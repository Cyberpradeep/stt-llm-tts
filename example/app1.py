from google.genai import Client
from pipecat.frames.frames import EndFrame, CancelFrame
from session import BookingSession
import asyncio
from datetime import datetime, time, timedelta
import traceback
import re
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import StartFrame
from google.genai.types import HarmCategory, HarmBlockThreshold, ProactivityConfig
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    InputParams,
    GeminiModalities,
    GeminiVADParams,
    ContextWindowCompressionParams,
)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleThinkingConfig
from contextlib import asynccontextmanager, nullcontext
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame, TranscriptionFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import InputAudioRawFrame, AudioRawFrame as _AudioRawFrame, InputTextRawFrame
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import TextFrame
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import time as time_module
# import GeminiLiveWebsocketTransport, GeminiLiveServiceOptions from
import json
# from pipecat.audio.vad.vad_analyzer import VADParams
from models import db, Patient, Doctor, Appointment
from flask import Flask
from flask_cors import CORS
from pipecat.frames.frames import MetricsFrame, EndFrame, CancelFrame
import time as time_module
from google.genai import types
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from lmnr import Laminar
from dotenv import load_dotenv


import os
import uuid
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from pipecat.utils.tracing.setup import setup_tracing

load_dotenv()

lmnr_project_key = os.getenv("LMNR_PROJECT_API_KEY")
lmnr_base_url = os.getenv("LMNR_BASE_URL")
lmnr_otlp_endpoint = os.getenv(
    "LMNR_OTLP_ENDPOINT", "https://api.lmnr.ai:8443")
tracing_enabled = False
exporter = None

if lmnr_project_key:
    exporter = OTLPSpanExporter(
        endpoint=lmnr_otlp_endpoint,
        headers={
            "authorization": f"Bearer {lmnr_project_key}"
        },
    )

    setup_tracing(
        service_name="priya-hospital-bot",
        exporter=exporter,
        console_export=False,  # Set True locally to debug
    )

    if lmnr_base_url:
        Laminar.initialize(base_url=lmnr_base_url)
    else:
        Laminar.initialize()

    tracing_enabled = True
else:
    print("LMNR_PROJECT_API_KEY not set; Laminar tracing disabled.")

task = None
greeted = False
context_aggregator = None
transcript_client: list = []
generat_ui: list = []
ui_list = []
llm = None
session = BookingSession()


def _laminar_set_trace_session_id(session_id: str) -> None:
    if not tracing_enabled:
        return
    try:
        Laminar.set_trace_session_id(session_id)
    except AttributeError:
        try:
            Laminar.setTraceSessionId(session_id)
        except Exception:
            return


def _laminar_set_trace_metadata(metadata: dict) -> None:
    if not tracing_enabled or not metadata:
        return
    try:
        Laminar.set_trace_metadata(metadata)
    except AttributeError:
        try:
            Laminar.setTraceMetadata(metadata)
        except Exception:
            return


def _laminar_set_span_attributes(attributes: dict) -> None:
    if not tracing_enabled or not attributes:
        return
    try:
        Laminar.set_span_attributes(attributes)
    except Exception:
        return


def _laminar_session_span(name: str):
    if not tracing_enabled:
        return nullcontext()
    try:
        return Laminar.start_as_current_span(name)
    except Exception:
        return nullcontext()

# class SessionMetricsAccumulator(FrameProcessor):
#     def __init__(self, context, system_instruction, api_key):
#         super().__init__()
#         self.context = context
#         self.system_instruction = system_instruction
#         self.start_time = time_module.time()
#         self.rest_client = Client(api_key=api_key)
#         self.bot_start_time = self.start_time
#         self.bot_end_time = self.start_time

#     def mark_bot_started(self):
#         self.bot_start_time = time_module.time()

#     def mark_bot_stopped(self):
#         self.bot_end_time = time_module.time()

#     async def process_frame(self, frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, (EndFrame, CancelFrame)):
#             duration = time_module.time() - self.start_time
#             print("\n" + "="*50)
#             print(" SESSION ENDED - CALCULATING API BILLING...")

#             # time_taken=self.bot_end_time - self.bot_start_time
#             time_taken=self.bot_start_time-self.bot_end_time
#             print(f"time taken : {time_taken}")
#             bot_tokens=int(time_taken * 25)
#             print(f"bot time token: {bot_tokens}")
#             print(f"idle tokens: {int(duration*32)}")
#             audio_tokens = int(duration * 32)+bot_tokens

#             text_tokens = 0
#             try:
#                 num_turns = 0
#                 full_history_text = ""

#                 for msg in self.context.messages:
#                     if "content" in msg and msg["content"]:
#                         full_history_text += f"{msg['role']}: {msg['content']}\n"
#                         num_turns += 1

#                 if num_turns > 0:
#                     sys_res = self.rest_client.models.count_tokens(
#                         model='gemini-2.5-flash',
#                         contents=self.system_instruction
#                     )
#                     sys_tokens = sys_res.total_tokens

#                     msg_res = self.rest_client.models.count_tokens(
#                         model='gemini-2.5-flash',
#                         contents=full_history_text
#                     )
#                     total_msg_tokens = msg_res.total_tokens

#                     cumulative_sys_cost = sys_tokens * num_turns

#                     cumulative_msg_cost = total_msg_tokens * ((num_turns + 1) / 2)

#                     text_tokens = int(cumulative_sys_cost + cumulative_msg_cost)

#             except Exception as e:
#                 print(f"Failed to calculate API tokens: {e}")

#             final_total = audio_tokens + text_tokens

#             print(f"  └ Audio Estimate (Duration x 57): {audio_tokens}")
#             print(f"  └ Text/Context (Official API):    {text_tokens}")
#             print("-" * 50)
#             print(f"Total Call Duration:  {duration:.2f} seconds")
#             print(f"Final Total Tokens:   {final_total}")
#             print("="*50 + "\n")

#         await self.push_frame(frame, direction)


# class SessionMetricsAccumulator(FrameProcessor):
#     def __init__(self, context, system_instruction, api_key):
#         super().__init__()
#         self.context = context
#         self.system_instruction = system_instruction
#         self.start_time = time_module.time()
#         self.rest_client = Client(api_key=api_key)

#         # We need an accumulator to add up EVERY time she speaks!
#         self.bot_start_time = 0
#         self.total_bot_speaking_time = 0

#     def mark_bot_started(self):
#         self.bot_start_time = time_module.time()

#     def mark_bot_stopped(self):
#         # Calculate the duration of THIS specific sentence, and ADD it to the total
#         if self.bot_start_time > 0:
#             self.total_bot_speaking_time += (time_module.time() - self.bot_start_time)
#             self.bot_start_time = 0 # Reset for her next sentence

#     async def process_frame(self, frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)

#         if isinstance(frame, (EndFrame, CancelFrame)):
#             duration = time_module.time() - self.start_time
#             print("\n" + "="*50)
#             print("📞 SESSION ENDED - CALCULATING API BILLING...")

#             # ========================================================
#             # 1. AUDIO MATH: Total Time * 32 + Bot Time * 25
#             # ========================================================
#             input_tokens = int(duration * 32)
#             bot_tokens = int(self.total_bot_speaking_time * 25)
#             audio_tokens = input_tokens + bot_tokens

#             idle_time = max(0, duration - self.total_bot_speaking_time)

#             # ========================================================
#             # 2. TEXT MATH: count_tokens + Arithmetic Progression
#             # ========================================================
#             text_tokens = 0
#             try:
#                 num_turns = 0
#                 full_history_text = ""

#                 for msg in self.context.messages:
#                     if "content" in msg and msg["content"]:
#                         full_history_text += f"{msg['role']}: {msg['content']}\n"
#                         num_turns += 1

#                 if num_turns > 0:
#                     sys_res = self.rest_client.models.count_tokens(
#                         model='gemini-2.5-flash',
#                         contents=self.system_instruction
#                     )
#                     sys_tokens = sys_res.total_tokens

#                     msg_res = self.rest_client.models.count_tokens(
#                         model='gemini-2.5-flash',
#                         contents=full_history_text
#                     )
#                     total_msg_tokens = msg_res.total_tokens

#                     cumulative_sys_cost = sys_tokens * num_turns
#                     cumulative_msg_cost = total_msg_tokens * ((num_turns + 1) / 2)

#                     text_tokens = int(cumulative_sys_cost + cumulative_msg_cost)

#             except Exception as e:
#                 print(f"Failed to calculate API tokens: {e}")

#             final_total = audio_tokens + text_tokens

#             print(f"  └ Audio Input (Duration x 32):  {input_tokens}")
#             print(f"  └ Audio Output (Priya x 25):    {bot_tokens} (Bot spoke for {self.total_bot_speaking_time:.2f}s)")
#             print(f"  └ Text/Context (Official API):  {text_tokens}")
#             print("-" * 50)
#             print(f"Total Call Duration:  {duration:.2f} seconds")
#             print(f"Final Total Tokens:   {final_total}")
#             print(f"Idle Mic/User Time:   {idle_time:.2f} seconds")
#             print("="*50 + "\n")

#         await self.push_frame(frame, direction)

# # class SessionMetricsAccumulator(FrameProcessor):
# #     def __init__(self, context):
# #         super().__init__()
# #         self.context = context
# #         self.start_time = time_module.time()

# #     async def process_frame(self, frame, direction: FrameDirection):
# #         await super().process_frame(frame, direction)

# #         if isinstance(frame, (EndFrame, CancelFrame)):
# #             end_time = time_module.time()
# #             total_seconds = end_time - self.start_time

# #             estimated_audio_tokens = int(total_seconds * 32)

# #             print("\n" + "="*50)
# #             print("SESSION ENDED - FINAL AUDIO BILLING")
# #             print(f"Total Call Duration:    {total_seconds:.2f} seconds")
# #             print(f"Estimated Audio Tokens: {estimated_audio_tokens}")
# #             print("="*50 + "\n")


# #         await self.push_frame(frame, direction)


class SessionMetricsAccumulator(FrameProcessor):
    def __init__(self, context, system_instruction, api_key):
        super().__init__()
        self.context = context
        self.system_instruction = system_instruction
        self.start_time = time_module.time()
        self.rest_client = Client(api_key=api_key)

        self.bot_start_time = 0
        self.total_bot_speaking_time = 0

    def mark_bot_started(self):
        self.bot_start_time = time_module.time()

    def mark_bot_stopped(self):
        if self.bot_start_time > 0:
            self.total_bot_speaking_time += (time_module.time() -
                                             self.bot_start_time)
            self.bot_start_time = 0

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (EndFrame, CancelFrame)):
            duration = time_module.time() - self.start_time
            print("\n" + "=" * 50)
            print("📞 SESSION ENDED - CALCULATING API BILLING...")

            # ============================================================
            # 1. AUDIO TOKENS
            #    Input  : entire session wall-clock time @ 32 tok/s
            #             (silence is billed the same as speech per Google)
            #    Output : only bot speaking time @ 25 tok/s
            # ============================================================
            input_tokens = int(duration * 32)
            bot_tokens = int(self.total_bot_speaking_time * 25)
            audio_tokens = input_tokens + bot_tokens

            idle_time = max(0, duration - self.total_bot_speaking_time)

            # ============================================================
            # 2. TEXT / CONTEXT-WINDOW TOKENS
            #    Live API charges cumulatively:
            #      Turn 1 pays: sys + msg1
            #      Turn 2 pays: sys + msg1 + msg2
            #      Turn N pays: sys + msg1 + ... + msgN
            #    So we rebuild the growing context window per turn and
            #    call count_tokens each time.
            # ============================================================
            text_tokens = 0
            try:
                messages = [
                    msg for msg in self.context.messages
                    if "content" in msg and msg["content"]
                ]
                print(f"messages for token counting: {messages}")
                num_turns = len(messages)

                if num_turns > 0:
                    # System prompt is re-sent every turn — count it once
                    sys_res = self.rest_client.models.count_tokens(
                        model='gemini-2.5-flash',
                        contents=self.system_instruction
                    )
                    sys_tokens = sys_res.total_tokens
                    print("System instruction tokens:", sys_tokens)

                    # Walk through turns, growing the history each time
                    cumulative_text_cost = 0
                    running_history = ""

                    for msg in messages:
                        running_history += f"{msg['role']}: {msg['content']}\n"
                        msg_res = self.rest_client.models.count_tokens(
                            model='gemini-2.5-flash',
                            contents=running_history
                        )
                        # This turn's context window = system + history so far
                        cumulative_text_cost += sys_tokens + msg_res.total_tokens
                        print(
                            f"Turn {num_turns} tokens (sys + history up to this turn): {sys_tokens + msg_res.total_tokens}")

                    text_tokens = int(cumulative_text_cost)

            except Exception as e:
                print(f"Failed to calculate text tokens: {e}")

            final_total = audio_tokens + text_tokens

            print(
                f"  └ Audio Input  (session {duration:.1f}s × 32):          {input_tokens} tokens")
            print(
                f"  └ Audio Output (bot {self.total_bot_speaking_time:.1f}s × 25):             {bot_tokens} tokens")
            print(
                f"  └ Text/Context (cumulative window, {num_turns} turns):  {text_tokens} tokens")
            print("-" * 50)
            print(
                f"Total Call Duration : {duration:.2f}s  (idle: {idle_time:.2f}s)")
            print(f"Final Total Tokens  : {final_total}")
            print("=" * 50 + "\n")

        await self.push_frame(frame, direction)


class RawPCMSerializer(FrameSerializer):

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self._sample_rate = sample_rate

    async def serialize(self, frame) -> bytes | None:
        if isinstance(frame, _AudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: bytes | str):
        if not isinstance(data, bytes) or len(data) == 0:
            return None
        return InputAudioRawFrame(
            audio=data,
            sample_rate=self._sample_rate,
            num_channels=1,
        )


class LaminarMetricsBridge(FrameProcessor):
    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for metric in frame.data:
                if isinstance(metric, LLMUsageMetricsData):
                    attrs = {
                        "llm.prompt_tokens": metric.value.prompt_tokens,
                        "llm.completion_tokens": metric.value.completion_tokens,
                        "llm.total_tokens": metric.value.total_tokens,
                    }
                    if metric.model:
                        attrs["llm.model"] = metric.model
                    if metric.value.cache_read_input_tokens is not None:
                        attrs["llm.cache_read_input_tokens"] = metric.value.cache_read_input_tokens
                    if metric.value.cache_creation_input_tokens is not None:
                        attrs["llm.cache_creation_input_tokens"] = metric.value.cache_creation_input_tokens
                    if metric.value.reasoning_tokens is not None:
                        attrs["llm.reasoning_tokens"] = metric.value.reasoning_tokens

                    _laminar_set_span_attributes(attrs)

        await self.push_frame(frame, direction)


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = ''
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
CORS(app, origins="*", supports_credentials=False)


class TokenCheck(FrameProcessor):
    async def token_check(self, frame, direction: FrameDirection):
        if isinstance(frame, TextFrame):
            if "<ctrl" in frame.text:
                return
            await self.push_frame(frame, direction)


async def delay_tool(p, result, call_time: float, min_delay: float = 2.0, after_delay: float = 0.1):
    elapsed = time_module.time() - call_time
    remaining = min_delay-elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)
    await p.result_callback(result)
    await asyncio.sleep(after_delay)


def book_appointment(name, age, gender, phone, doctor, date=None, boo_k_time=None, reason=None):
    with app.app_context():
        try:
            if date:
                for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                    try:
                        date = datetime.strptime(
                            date, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        pass
            print("booking appointment called")
            if not boo_k_time:
                return f"get the time for booking appointment"
            if not reason:
                return f"get the reason for booking appointment"
            if not date:
                return f"get the date for booking appointment"
            book_date = datetime.strptime(date, "%Y-%m-%d").date()

            for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                try:
                    book_time = datetime.strptime(
                        boo_k_time.strip(), fmt).time()
                    break
                except ValueError:
                    pass
            else:
                return f"Invalid time format: {boo_k_time}. Please provide time like 10:45 AM or 14:30"

            boo_k_time = book_time.strftime("%H:%M")
            clean_doctor = doctor.replace("Dr.", "").replace("Dr ", "").strip()
            doctor_name = Doctor.query.filter(
                Doctor.name.ilike(f"%{clean_doctor}%")).first()
            print(
                f"clean doctor name: {clean_doctor}, doctor found: {doctor_name.name if doctor_name else 'None'}")
            print("time :", boo_k_time)
            phone_exist = Patient.query.filter_by(phone=phone).first()

            if phone_exist and phone_exist.name != name:
                print("phone number exist")
                return f"Phone number {phone} is already exist for another patient, please provide a different phone number"

            if not doctor_name:
                return f"Doctor not found: {doctor}"

            if len(phone) < 10 or len(phone) > 10:
                return f"Phone number {phone} is not valid and ask the user once again tell your phone number "

            if book_date < datetime.now().date():
                return f"Date {book_date} is not valid the date has already passed"

            if doctor_name.work_brk_st_time <= book_time <= doctor_name.work_brk_end_time:
                return f"Doctor is on break during {boo_k_time}"

            if not (doctor_name.work_st_time <= book_time <= doctor_name.work_end_time):
                return f"Doctor is not available at {boo_k_time}"

            already_booked = Appointment.query.filter_by(
                doctor_id=doctor_name.id, date=book_date, time=book_time).first()

            if already_booked:
                return f"Doctor is already booked at {boo_k_time} on {date}"

            patient = Patient.query.filter_by(phone=phone).first()

            if patient:
                existing = Appointment.query.filter_by(
                    patient_id=patient.id,
                    doctor_id=doctor_name.id,
                    date=book_date,
                    time=book_time
                ).first()
                if existing:
                    return f"You have already booked an appointment with Dr. {doctor} on {date} at {boo_k_time}"

            if not patient:
                patient = Patient(
                    name=name,
                    age=age,
                    gender=gender,
                    phone=phone
                )
                db.session.add(patient)
                db.session.commit()

            new_apt = Appointment(
                patient_id=patient.id,
                doctor_id=doctor_name.id,
                date=book_date,
                time=book_time,
                reason=reason
            )
            db.session.add(new_apt)
            db.session.commit()

            print("booking confirmed")
            return {
                "status": "success",
                "data": {
                    "name": name,
                    "doctor": doctor_name.name,
                    "date": date,
                    "time": boo_k_time,
                    "reason": reason,
                    "appointment_id": new_apt.id,
                }
            }
        except Exception as e:
            print(f"booking appoint err: str{e}")
            db.session.rollback()
            return f"Error{str(e)}"


# def pre_appoint(phone):
#     with app.app_context():
#         try:
#             patient = Patient.query.filter_by(phone=phone).first()
#             if not patient:
#                 return f"No patient found with phone number: {phone}"
#             appoint = Appointment.query.filter_by(patient_id=patient.id).all()
#             if not appoint:
#                 return f"No previous appointments found for phone number: {phone}"
#             apt_list = []
#             for apt in appoint:
#                 doctor = db.session.query(Doctor).get(apt.doctor_id)
#                 print(
#                     f"Appointment with Dr. {doctor.name} on {apt.date} at {apt.time}, Reason: {apt.reason}, Patient name:{patient.name}")
#                 apt_list.append(
#                     f"Appointment with Dr. {doctor.name} on {apt.date} at {apt.time}, Reason: {apt.reason}, Patient name:{patient.name}")
#             return apt_list
#         except Exception as e:
#             print(f"error:{str(e)}")
#             return f"Error{str(e)}"


def available_slots_fun(doctor: str, date: str):
    with app.app_context():
        try:
            parsed_date = None
            for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                try:
                    parsed_date = datetime.strptime(date, fmt).date()
                    break
                except ValueError:
                    continue
            if not parsed_date:
                return "reformat the date with DD-MM-YYYY"
            date = parsed_date.strftime("%Y-%m-%d")

            def normalize_name(value: str) -> str:
                return re.sub(r"[^a-z0-9]+", "", value.lower())

            doc = Doctor.query.filter(Doctor.name.ilike(f"%{doctor}%")).first()
            if not doc:
                normalized = normalize_name(doctor)
                if normalized:
                    for candidate in Doctor.query.all():
                        if normalize_name(candidate.name) == normalized:
                            doc = candidate
                            break
            print(f"doctor name: {doctor}")
            if not doc:
                print(f"No doctor found with name: {doctor}")
                return f"No doctor found with name: {doctor}"
            booked = Appointment.query.filter_by(
                doctor_id=doc.id, date=date).all()
            print(f"booked slots for doctor {doctor} on date {date}: {booked}")
            avai_slot = []
            current_time = datetime.now().time()
            doc_strat = doc.work_st_time
            doc_end = doc.work_end_time
            doc_brk_strt = doc.work_brk_st_time
            doc_brk_end = doc.work_brk_end_time
            start = doc_strat.hour * 60 + doc_strat.minute
            end = doc_end.hour * 60 + doc_end.minute
            brk_start = doc_brk_strt.hour * 60 + doc_brk_strt.minute
            brk_end = doc_brk_end.hour * 60 + doc_brk_end.minute
            booked_slot = []
            for t in booked:
                booked_slot.append(t.time)
            print(
                f"start:{start},end:{end},brk_start:{brk_start},brk_end:{brk_end}")
            today = datetime.now().date()
            date_v = datetime.strptime(date, "%Y-%m-%d").date()
            for i in range(start, end, doc.meet_duration):
                hour = i//60
                minute = i % 60
                print(f"hour:{hour},minute:{minute}")
                slot_time = f"{hour:02d}:{minute:02d}"
                time_slt = time(hour, minute)
                if time_slt in booked_slot:
                    continue
                if i >= brk_start and i < brk_end:
                    continue
                if today == date_v and time_slt <= current_time:
                    continue
                avai_slot.append(slot_time)
                print(f"slot_time:{slot_time}, time_slt:{time_slt}")
            print(avai_slot)
            return avai_slot
        except Exception as e:
            print(f"available slot error: {str(e)}")
            return f"Error{str(e)}"


def appointment_fetch(phone):
    with app.app_context():
        try:
            if len(phone) != 10:
                print(f"Invalid phone number length: {phone}")
                return f"Invalid phone number length: {phone}"
            patient = Patient.query.filter_by(phone=phone).first()
            if not patient:
                return f"No patient found with phone number: {phone}"
            appoint = Appointment.query.filter_by(patient_id=patient.id).all()
            if not appoint:
                return f"No previous appointments found for phone number: {phone}"
            apt_list = []
            for apt in appoint:
                doctor = db.session.query(Doctor).get(apt.doctor_id)
                print(
                    f"Appointment with Dr. {doctor.name} on {apt.date} at {apt.time}, Reason: {apt.reason}, Patient name:{patient.name}")
            #     apt_list.append(
            #         f"Appointment with Dr. {doctor.name} on {apt.date} at {apt.time}, Reason: {apt.reason}, Patient name:{patient.name}")
            # return f" Ask the user to whether they want to meet the same doctor which you have previously visited if they say okay continue with this else go on with who they want to meet or for what reason , Previous Appointment {apt_list}"
                apt_list.append({
                    "doctor": doctor.name,
                    "date": apt.date,
                    "time": apt.time,
                    "reason": apt.reason
                })
                return apt_list
        except Exception as e:
            print(f"error:{str(e)}")
            return f"appointment fetch details Error{str(e)}"


def update_appointment(phone, appointment_id, **updates):
    with app.app_context():
        try:
            patient = Patient.query.filter_by(phone=phone).first()
            if not patient:
                return f"No patient found with phone number: {phone}"

            apt = Appointment.query.filter_by(
                id=appointment_id,
                patient_id=patient.id,
            ).first()
            if not apt:
                return f"No appointment found with id: {appointment_id}"

            now_dt = datetime.now()
            current_apt_dt = datetime.combine(apt.date, apt.time)
            if current_apt_dt < now_dt:
                return "Cannot update past appointment"

            old_date = apt.date
            old_time = apt.time

            target_date = apt.date
            target_time = apt.time
            target_doctor_id = apt.doctor_id

            if 'date' in updates and updates['date']:
                try:
                    target_date = datetime.strptime(
                        updates['date'], "%Y-%m-%d").date()
                except ValueError:
                    return f"Invalid date format: {updates['date']}. Please provide date like YYYY-MM-DD"

            if 'time' in updates and updates['time']:
                parsed_time = None
                for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                    try:
                        parsed_time = datetime.strptime(
                            updates['time'].strip(), fmt).time()
                        break
                    except ValueError:
                        pass
                if not parsed_time:
                    return f"Invalid time format: {updates['time']}. Please provide time like 10:45 AM or 14:30"
                target_time = parsed_time

            if 'reason' in updates and updates['reason']:
                apt.reason = updates['reason']

            if 'doctor' in updates and updates['doctor']:
                doc = Doctor.query.filter(
                    Doctor.name.ilike(f"%{updates['doctor']}%")
                ).first()
                if not doc:
                    return f"Doctor not found: {updates['doctor']}"
                target_doctor_id = doc.id

            target_dt = datetime.combine(target_date, target_time)
            if target_dt < now_dt:
                return "Cannot update to a passed date/time. Previous booking date and time are kept unchanged."

            target_doctor = db.session.get(Doctor, target_doctor_id)
            if not target_doctor:
                return "Doctor details not found for update"

            if target_doctor.work_brk_st_time <= target_time <= target_doctor.work_brk_end_time:
                return f"Doctor is on break during {target_time.strftime('%H:%M')}. Previous booking date and time are kept unchanged."

            if not (target_doctor.work_st_time <= target_time <= target_doctor.work_end_time):
                return f"Doctor is not available at {target_time.strftime('%H:%M')}. Previous booking date and time are kept unchanged."

            already_booked = Appointment.query.filter(
                Appointment.doctor_id == target_doctor_id,
                Appointment.date == target_date,
                Appointment.time == target_time,
                Appointment.id != apt.id,
            ).first()
            if already_booked:
                return f"Doctor is already booked at {target_time.strftime('%H:%M')} on {target_date}. Previous booking date and time are kept unchanged."

            apt.date = target_date
            apt.time = target_time
            apt.doctor_id = target_doctor_id

            if 'name' in updates and updates['name']:
                patient.name = updates['name']
            if 'age' in updates and updates['age']:
                patient.age = updates['age']
            if 'gender' in updates and updates['gender']:
                patient.gender = updates['gender']

            db.session.commit()
            doctor = db.session.get(Doctor, apt.doctor_id)
            return {
                'status': 'success',
                'message': f"Appointment updated successfully",
                'data': {
                    'name': patient.name,
                    'age': patient.age,
                    'gender': patient.gender,
                    'phone': phone,
                    'doctor': doctor.name,
                    'date': str(apt.date),
                    'time': str(apt.time)[:5],
                    'reason': apt.reason,
                    'appointment_id': apt.id,
                    'previous_date': str(old_date),
                    'previous_time': str(old_time)[:5]
                }
            }
        except Exception as e:
            db.session.rollback()
            print(f"update appointment err: {str(e)}")
            return f"Error: {str(e)}"


def fetch_upcoming_appointments(phone):
    with app.app_context():
        try:
            patient = Patient.query.filter_by(phone=phone).first()
            if not patient:
                return []
            apts = Appointment.query.filter_by(patient_id=patient.id).all()
            upcoming = []
            today = datetime.now().date()
            for apt in apts:
                if apt.date >= today:
                    doctor = db.session.query(Doctor).get(apt.doctor_id)
                    upcoming.append({
                        'id': apt.id,
                        'doctor': doctor.name,
                        'date': str(apt.date),
                        'time': str(apt.time)[:5],
                        'reason': apt.reason,
                        "name": patient.name,
                        "age": patient.age,
                        "gender": patient.gender
                    })
            return upcoming
        except Exception as e:
            print(f"fetch upcoming err: {str(e)}")
            return []


def cancel_appointment(phone, appointment_id):
    with app.app_context():
        try:
            patient = Patient.query.filter_by(phone=phone).first()
            if not patient:
                return f"No patient found with phone number: {phone}"

            apt = Appointment.query.filter_by(
                id=appointment_id,
                patient_id=patient.id,
            ).first()
            if not apt:
                return f"No appointment found with id: {appointment_id}"

            apt_dt = datetime.combine(apt.date, apt.time)
            if apt_dt < datetime.now():
                return "Cannot cancel past appointment"

            doctor = db.session.get(Doctor, apt.doctor_id)
            cancelled_data = {
                'appointment_id': apt.id,
                'name': patient.name,
                'phone': patient.phone,
                'doctor': doctor.name if doctor else "Unknown",
                'date': str(apt.date),
                'time': str(apt.time)[:5],
                'reason': apt.reason,
            }

            db.session.delete(apt)
            db.session.commit()

            return {
                'status': 'success',
                'message': 'Appointment cancelled successfully',
                'data': cancelled_data,
            }
        except Exception as e:
            db.session.rollback()
            print(f"cancel appointment err: {str(e)}")
            return f"Error: {str(e)}"


def get_appointment_for_cancel(phone, appointment_id):
    with app.app_context():
        try:
            patient = Patient.query.filter_by(phone=phone).first()
            if not patient:
                return None, f"No patient found with phone number: {phone}"

            apt = Appointment.query.filter_by(
                id=appointment_id,
                patient_id=patient.id,
            ).first()
            if not apt:
                return None, f"No appointment found with id: {appointment_id}"

            apt_dt = datetime.combine(apt.date, apt.time)
            if apt_dt < datetime.now():
                return None, "Cannot cancel past appointment"

            doctor = db.session.get(Doctor, apt.doctor_id)
            return {
                'appointment_id': apt.id,
                'name': patient.name,
                'phone': patient.phone,
                'doctor': doctor.name if doctor else "Unknown",
                'date': str(apt.date),
                'time': str(apt.time)[:5],
                'reason': apt.reason,
            }, None
        except Exception as e:
            print(f"get appointment for cancel err: {str(e)}")
            return None, f"Error: {str(e)}"


def time_format_conversion(time_str):
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
        try:
            parsed_time = datetime.strptime(time_str.strip(), fmt).time()
            return parsed_time.strftime("%H:%M")
        except ValueError:
            return f"Invalid time format: {time_str}. Please provide time like 10:45 AM or 14:30"


def date_time_validation(date_str, time_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return f"Invalid date format: {date_str}. Please provide date like YYYY-MM-DD"

    time_conversion_result = time_format_conversion(time_str)
    if isinstance(time_conversion_result, str) and time_conversion_result.startswith("Invalid time format"):
        return time_conversion_result

    time_obj = datetime.strptime(time_conversion_result, "%H:%M").time()

    now = datetime.now()
    if date_obj < now.date() or (date_obj == now.date() and time_obj <= now.time()):
        return "That date and time has already passed. Please provide a valid upcoming date and time."

    return "Valid"


def validate_phone_number(phone):
    if len(phone) != 10 or not phone.isdigit():
        return f"That phone number doesn't look right. Could you say it again?"
    return "Valid"


def clinic_info():
    return {
        'clinic': 'city clinic',
        'available-day': 'Monday-Saturday',
        'available-time': '10 AM - 8 PM',
        'break-time': '12:30 PM - 1:30 PM',
        'lunch-break': '12:30 PM - 1:30 PM',
        'slot-duration': '15 minutes',
    }


def doctor_recommendation(age, reason):
    if age <= 15:
        return {
            "Doctor": "Dr. Ramya",
            "Specialty": "Pediatrician",
        }
    else:
        return {
            "Doctor": "Dr. Shakthi",
            "Specialty": "General Physician",
        }


def doctor_info(doctor_name):
    doctor = Doctor.query.filter(Doctor.name.ilike(f"%{doctor_name}%")).first()
    if doctor:
        return {
            "Doctor": doctor.name,
            "Specialty": doctor.specialty,
        }
    else:
        return f"No doctor found with name: {doctor_name} and don't use name starts with Dr. use name only"


# system_instruction = """
# You are Priya, a warm receptionist at City Clinic.

# PERSONA: Warm, natural, under 20 words per response. Never robotic. Show empathy. Never reveal you're an AI.

# SCOPE: Only appointments. Anything else: "I can only help with City Clinic appointments."

# LANGUAGE: English default. Match Tamil/Tanglish if user uses it.

# GREETING (once only): "Hello, I'm Priya from City Clinic. How can I help you today?"

# DOCTORS: Dr. Shakthi (age >15). Dr. Ramya (age ≤15). Hours: 10AM–9PM Mon–Sat, break 12:30–1:30PM.

# FIRST ACTION: Call set_intent the moment intent is clear. Before anything else.

# BATCH RULE — CRITICAL:
# STATE tells you what to ask via the "ask" field. Follow it exactly:
# - ask_name_age_gender → ONE sentence: "Could I get your name, age, and gender?"
# - ask_age_gender → "And your age and gender?"
# - ask_reason_and_date → "What's the reason for your visit, and when would you like to come?"
# - ask_phone_then_fetch → Ask phone. Then silently call appointment_fetch.
# - call_available_slots_then_ask_slot → Say "One second." Call available_slots. Read slots. Ask patient to pick.
# - call_show_confirmation → Call show_confirmation with all collected details now.
# - call_fetch_upcoming → Silently call fetch_upcoming_appointments now.
# - call_cancel_confirmed_false → Call cancel_appointment with confirmed=false now.
# - waiting_user_yes_to_book → Ask "Shall I confirm this booking?"
# - waiting_user_yes_to_update → Ask "Shall I confirm this update?"
# - waiting_user_yes_to_cancel → Ask "Are you sure you want to cancel?"
# - ask_which_appointment_to_update → "Which appointment would you like to update?" (shown on screen)
# - ask_which_appointment_to_cancel → "Which appointment would you like to cancel?" (shown on screen)
# - ask_what_to_change_then_show_confirmation → Ask what they want to change. Collect. Call show_confirmation.

# STATE RULE:
# After each user message you receive a [STATE] block.
# - "got"= already collected. NEVER re-ask any field listed here.
# - "ask"= your ONLY next action. Do exactly this and nothing else.
# - "f" = status flags. Use to understand current progress.

# TOOLS:
# - set_intent → silent. call when intent clear.
# - appointment_fetch → silent always.
# - available_slots → say "One second." then call. NEVER suggest times without this.
# - show_confirmation → always before book_appointment or update_appointment.
# - book_appointment → only after show_confirmation and user says yes.
# - fetch_upcoming_appointments → silent.
# - update_appointment → only after show_confirmation and user says yes.
# - cancel_appointment → confirmed=false first, confirmed=true only after user says yes.

# Follow ns field in every tool result exactly.
# TIME: 12-hour AM/PM always.
# """

system_instruction = """You are Priya, City Clinic receptionist. Be warm and brief (<20 words per reply). Never reveal you are AI. Scope: appointments only. Else say: "I can only help with City Clinic appointments."
LANGUAGE: English default. Mirror Tamil/Tanglish if patient uses it.
GREETING (once only): "Hello, I'm Priya from City Clinic. How can I help you today?"
DOCTORS: Dr. Shakthi (age >15), Dr. Ramya (age ≤15). Hours: 10AM–9PM Mon–Sat, break 12:30–1:30PM.
TIME: 12-hr AM/PM. DATE: DD-MM-YYYY.
BOOKING FLOW — collect in this exact order:
1. Name
2. Age
3. Gender
4. Phone (10 digits)
5. Reason for visit
6. Preferred date → auto-assign doctor by age rule
7. Show available slots verbatim from tool. Never add extra slots. Wait for selection.
8. CONFIRM — read back: User detaiks: "Shall I confirm this booking?"
9. On yes → call booking tool. On no → ask what to correct.

SLOTS: Repeat tool output exactly. Do not invent or add slots."""


safety_settings = [
    {
        "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
    {
        "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        "threshold": HarmBlockThreshold.BLOCK_NONE,
    },
]


tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="set_intent",
                description="Set booking intent (book/update/cancel).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"intent": types.Schema(type=types.Type.STRING, enum=[
                                                       "book", "update", "cancel"])},
                    required=["intent"]
                )
            ),
            types.FunctionDeclaration(
                name="book_appointment",
                description="Save appointment to database.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name":   types.Schema(type=types.Type.STRING),
                        "age":    types.Schema(type=types.Type.INTEGER),
                        "gender": types.Schema(type=types.Type.STRING),
                        "phone":  types.Schema(type=types.Type.STRING),
                        "doctor": types.Schema(type=types.Type.STRING),
                        "date":   types.Schema(type=types.Type.STRING),
                        "time":   types.Schema(type=types.Type.STRING),
                        "reason": types.Schema(type=types.Type.STRING),
                    },
                    required=["name", "age", "gender", "phone",
                              "doctor", "date", "time", "reason"]
                )
            ),
            types.FunctionDeclaration(
                name="available_slots",
                description="Fetch open slots for a doctor on a date.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "doctor": types.Schema(type=types.Type.STRING),
                        "date":   types.Schema(type=types.Type.STRING),
                    },
                    required=["doctor", "date"]
                )
            ),
            types.FunctionDeclaration(
                name="show_confirmation",
                description="Show appointment details card to patient for confirmation.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "name":           types.Schema(type=types.Type.STRING),
                        "age":            types.Schema(type=types.Type.INTEGER),
                        "gender":         types.Schema(type=types.Type.STRING),
                        "phone":          types.Schema(type=types.Type.STRING),
                        "doctor":         types.Schema(type=types.Type.STRING),
                        "date":           types.Schema(type=types.Type.STRING),
                        "time":           types.Schema(type=types.Type.STRING),
                        "reason":         types.Schema(type=types.Type.STRING),
                        "appointment_id": types.Schema(type=types.Type.INTEGER),
                    },
                    required=["name", "age", "gender", "phone",
                              "doctor", "date", "time", "reason"]
                )
            ),
            types.FunctionDeclaration(
                name="appointment_fetch",
                description="Fetch patient's last appointment by phone number.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"phone": types.Schema(type=types.Type.STRING)},
                    required=["phone"]
                )
            ),
            types.FunctionDeclaration(
                name="fetch_upcoming_appointments",
                description="Fetch upcoming appointments for update or cancel flow.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={"phone": types.Schema(type=types.Type.STRING)},
                    required=["phone"]
                )
            ),
            types.FunctionDeclaration(
                name="update_appointment",
                description="Update an existing appointment in the database.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "phone":          types.Schema(type=types.Type.STRING),
                        "appointment_id": types.Schema(type=types.Type.INTEGER),
                        "name":           types.Schema(type=types.Type.STRING),
                        "age":            types.Schema(type=types.Type.INTEGER),
                        "gender":         types.Schema(type=types.Type.STRING),
                        "doctor":         types.Schema(type=types.Type.STRING),
                        "date":           types.Schema(type=types.Type.STRING),
                        "time":           types.Schema(type=types.Type.STRING),
                        "reason":         types.Schema(type=types.Type.STRING),
                    },
                    required=["phone", "appointment_id"]
                )
            ),
            types.FunctionDeclaration(
                name="cancel_appointment",
                description="Cancel appointment. confirmed=false to preview, confirmed=true to execute.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "phone":          types.Schema(type=types.Type.STRING),
                        "appointment_id": types.Schema(type=types.Type.INTEGER),
                        "confirmed":      types.Schema(type=types.Type.BOOLEAN),
                    },
                    required=["phone", "appointment_id"]
                )
            ),
        ]
    )
]


# async def book_appointment_call(p):
#     llm.set_audio_input_paused(True)
#     try:
#         a = p.arguments
#         res = await asyncio.to_thread(
#             book_appointment,
#             a["name"], a['age'], a['gender'], a['phone'], a['doctor'], a['date'], a['time'], a['reason']
#         )
#         if isinstance(res, dict) and res['status'] == 'success':
#             await gen_ui("summary", res['data'])
#             # await p.result_callback(res['message'])
#             # await p.result_callback(
#             #     f"BOOKING CONFIRMED IN DATABASE. "
#             #     f"Appointment ID created. "
#             #     f"Now tell patient: appointment booked for {res['data']['name']} "
#             #     f"with Dr.{res['data']['doctor']} on {res['data']['date']} "
#             #     f"at {res['data']['time']}. "
#             #     f"DO NOT say booked unless you see this message."
#             # )

#             print(f"booking confirmed name: {res['data']['name']}, doctor: {res['data']['doctor']}, date: {res['data']['date']}, time: {res['data']['time']}")

#             await p.result_callback(
#                 {
#                     "message":"Booking confirmed in db",
#                     "details": {
#                         "name": res['data']['name'],
#                         "doctor": res['data']['doctor'],
#                         "date": res['data']['date'],
#                         "time": res['data']['time']
#                     },
#                 }
#             )
#         else:
#             await p.result_callback(str(res))
#     except Exception as e:
#         await p.result_callback(f"Booking failed: {str(e)}")
#     finally:
#         llm.set_audio_input_paused(False)


# # async def previous_appointment_call(p):
# #     llm.set_audio_input_paused(True)
# #     try:
# #         res = await asyncio.to_thread(pre_appoint, p.arguments['phone'])
# #         # await gen_ui("previous", {
# #         #     'appointments': res if isinstance(res, list) else []
# #         # })
# #         await p.result_callback(res)
# #     except Exception as e:
# #         await p.result_callback(f"Error fetching appointments: {str(e)}")
# #     finally:
# #         llm.set_audio_input_paused(False)


# async def available_slots_call(p):
#     llm.set_audio_input_paused(True)
#     try:
#         a = p.arguments
#         res = await asyncio.to_thread(available_slots_fun, a['doctor'], a['date'])
#         # await gen_ui("avaislots", {
#         #     'doctor': a['doctor'],
#         #     'date': a['date'],
#         #     'slots': res if isinstance(res, list) else []
#         # })
#         if isinstance(res, list) and len(res) > 0:
#             def fmt_slot(s):
#                 h, m = s.split(":")
#                 hour = int(h)
#                 ampm = "AM" if hour < 12 else "PM"
#                 h12 = hour % 12 or 12
#                 return f"{h12}:{m} {ampm}"

#             voice_slots = ", ".join(fmt_slot(s) for s in res[:3])
#             prom_res = ({
#                 "slots": voice_slots,
#                 "note": "Offer these first. Ask if another preferred time."
#             })
#         elif isinstance(res, list) and len(res) == 0:
#             prom_res = "No slots available on that date. Ask patient for another date."
#         else:
#             prom_res = str(res)
#         await p.result_callback(prom_res)
#     except Exception as e:
#         await p.result_callback(f"Error fetching slots: {str(e)}")
#     finally:
#         llm.set_audio_input_paused(False)


# async def confirm_booking_call(p):
#     llm.set_audio_input_paused(True)
#     a = p.arguments
#     # Update session with everything collected
#     if a.get("name"):
#         session.patient.name = a["name"]
#     if a.get("age"):
#         session.patient.age = a["age"]
#     if a.get("gender"):
#         session.patient.gender = a["gender"]
#     if a.get("phone"):
#         session.patient.phone = a["phone"]
#     if a.get("doctor"):
#         session.patient.doctor = a["doctor"]
#     if a.get("date"):
#         session.patient.date = a["date"]
#     if a.get("time"):
#         session.patient.slot = a["time"]
#     if a.get("reason"):
#         session.patient.reason = a["reason"]
#     if a.get("appointment_id"):
#         session.patient.appointment_id = a["appointment_id"]

#     session.awaiting_confirmation = True           # ← waiting for user yes

#     next_action = (
#         "call update_appointment after user confirms"
#         if session.patient.appointment_id
#         else "call book_appoinftment after user confirms"
#     )
#     print(f"[CONFIRM BOOKING CALL] Next action: {next_action}")
#     await p.result_callback({
#         "message": "confirmation card shown to user. read details and ask yes or no.",
#         "details": a,
#         "next_step": next_action                   # ← inject next step
#     })
#     llm.set_audio_input_paused(False)

# # async def confirm_booking_call(p):
# #     llm.set_audio_input_paused(True)
# #     a = p.arguments
# #     await p.result_callback(
# #         {
# #             "message":"confirmation details",
# #             "name": a["name"],
# #             "age": a["age"],
# #             "gender":a["gender"],
# #             "phone": a["phone"],
# #             "doctor": a["doctor"],
# #             "date": a["date"],
# #             "time": a["time"],
# #             "reason": a["reason"],
# #             "function call":{
# #                 "if update": "call update_appointment with appointment_id",
# #                 "if new": "call book_appointment"
# #             }

# #         }
# #     )
# #     llm.set_audio_input_paused(False)


# async def appointment_fetch_call(p):
#     llm.set_audio_input_paused(True)
#     # llm.set_audio_output_paused(True)
#     try:
#         res = await asyncio.to_thread(appointment_fetch, p.arguments['phone'])
#         print(f"fetch appointment result: {res}")
#         await p.result_callback(res)
#     except Exception as e:
#         print(f"Error fetching patient history: {str(e)}")
#         await p.result_callback(f"Error fetching patient history: {str(e)}")
#     finally:
#         llm.set_audio_input_paused(False)
#         # llm.set_audio_output_paused(False)


# # async def fetch_upcoming_call(p):
# #     llm.set_audio_input_paused(True)
# #     try:
# #         res = await asyncio.to_thread(fetch_upcoming_appointments, p.arguments['phone'])
# #         await gen_ui("upcoming", {'appointments': res, 'phone': p.arguments['phone']})
# #         if res:
# #             lines = []
# #             for apt in res:
# #                 h, m = apt['time'].split(":")
# #                 hour = int(h)
# #                 ampm = "AM" if hour < 12 else "PM"
# #                 h12 = hour % 12 or 12
# #                 lines.append(
# #                     f"Appointment ID {apt['id']}: Dr.{apt['doctor']} on {apt['date']} "
# #                     f"at {h12}:{m} {ampm}, reason: {apt['reason']}"
# #                 )
# #             apt_text = " | ".join(lines)
# #             # await p.result_callback(
# #             #     f"Upcoming appointments found: {apt_text}. "
# #             #     f"Read these to the patient and ask which appointment they want to update or cancel. "
# #             #     f"IMPORTANT: When patient selects one, remember its Appointment ID. "
# #             #     f"For UPDATE — call available_slots to get new times, then call show_confirmation "
# #             #     f"with the SAME appointment_id, then call update_appointment (NOT book_appointment). "
# #             #     f"For CANCEL — call cancel_appointment with that appointment_id. "
# #             #     f"Never call book_appointment for an update — that creates a duplicate."
# #             # )
# #             await p.result_callback(
# #                 {
# #                     "appointments": apt_text
# #                 }
# #             )
# #         else:
# #             await p.result_callback("No upcoming appointments found for this patient.")
# #     except Exception as e:
# #         await p.result_callback(f"Error: {str(e)}")
# #     finally:
# #         llm.set_audio_input_paused(False)

# async def fetch_upcoming_call(p):
#     llm.set_audio_input_paused(True)
#     try:
#         res = await asyncio.to_thread(fetch_upcoming_appointments, p.arguments['phone'])
#         session.upcoming_shown = True                          # ← track state
#         await gen_ui("upcoming", {'appointments': res, 'phone': p.arguments['phone']})
#         if res:
#             lines = []
#             for apt in res:
#                 h, m = apt['time'].split(":")
#                 hour = int(h)
#                 ampm = "AM" if hour < 12 else "PM"
#                 h12 = hour % 12 or 12
#                 lines.append(
#                     f"ID {apt['id']}: Dr.{apt['doctor']} on {apt['date']} "
#                     f"at {h12}:{m} {ampm}, reason: {apt['reason']}"
#                 )
#             apt_text = " | ".join(lines)
#             print(f"Upcoming appointments: {apt_text}, next step: {session.update_next_step()}")
#             await p.result_callback({
#                 "appointments": apt_text,
#                 "next_step": session.update_next_step()        # ← inject next step
#             })
#         else:
#             await p.result_callback({
#                 "result": "No upcoming appointments found.",
#                 "next_step": "tell patient no appointments found"
#             })
#     except Exception as e:
#         print(f"Error fetching upcoming appointments: {str(e)}")
#         await p.result_callback(f"Error: {str(e)}")
#     finally:
#         llm.set_audio_input_paused(False)


# async def update_appointment_call(p):
#     llm.set_audio_input_paused(True)
#     try:
#         # Gate: block if show_confirmation was never called
#         if not session.awaiting_confirmation:
#             print(f"[UPDATE APPOINTMENT CALL] next step: {session.update_next_step()}")
#             await p.result_callback({
#                 "error": "show_confirmation must be called first.",
#                 "next_step": session.update_next_step()
#             })
#             return

#         a = p.arguments
#         res = await asyncio.to_thread(
#             update_appointment,
#             a['phone'], a['appointment_id'],
#             **{k: v for k, v in a.items() if k not in ['phone', 'appointment_id']}
#         )
#         if isinstance(res, dict) and res['status'] == 'success':
#             session.reset()                        # ← clean slate after success
#             await p.result_callback({
#                 "result": "Appointment updated successfully.",
#                 "details": res['data'],
#                 "next_step": "tell patient their appointment is updated. conversation done."
#             })
#         else:
#             print(f"Update failed: {res}, next step: {session.update_next_step()}")
#             await p.result_callback({
#                 "error": str(res),
#                 "next_step": "tell patient there was an issue. ask if they want to try again."
#             })
#     except Exception as e:
#         await p.result_callback(f"Update failed: {str(e)}")
#     finally:
#         llm.set_audio_input_paused(False)

# # async def update_appointment_call(p):
# #     llm.set_audio_input_paused(True)
# #     try:
# #         a = p.arguments
# #         res = await asyncio.to_thread(
# #             update_appointment,
# #             a['phone'], a['appointment_id'],
# #             **{k: v for k, v in a.items() if k not in ['phone', 'appointment_id']}
# #         )
# #         if isinstance(res, dict) and res['status'] == 'success':
# #             # await gen_ui("update_summary", res['data'])
# #             await p.result_callback(res['message'])
# #         else:
# #             await p.result_callback(str(res))
# #     except Exception as e:
# #         await p.result_callback(f"Update failed: {str(e)}")
# #     finally:
# #         llm.set_audio_input_paused(False)


# async def cancel_appointment_call(p):
#     llm.set_audio_input_paused(True)
#     try:
#         a = p.arguments
#         confirmed = bool(a.get('confirmed', False))
#         if not confirmed:
#             details, err = await asyncio.to_thread(get_appointment_for_cancel, a['phone'], a['appointment_id'])
#             if err:
#                 await p.result_callback(err)
#             else:
#                 # await gen_ui("cancel_confirm", details)
#                 await p.result_callback(f"ask user to confirm cancellation of appointment with Dr.{details['doctor']} on {details['date']} at {details['time']}. "
#                 )
#             return

#         res = await asyncio.to_thread(cancel_appointment, a['phone'], a['appointment_id'])
#         if isinstance(res, dict) and res.get('status') == 'success':
#             # await gen_ui("cancel_summary", res['data'])
#             # await p.result_callback(
#             #     f"CANCELLATION CONFIRMED IN DATABASE. "
#             #     f"Appointment {res['data']['appointment_id']} cancelled for {res['data']['name']} "
#             #     f"with Dr.{res['data']['doctor']} on {res['data']['date']} at {res['data']['time']}."
#             # )
#             await p.result_callback(
#                 {
#                     'status': 'success',
#                     'message': f"Appointment cancelled for {res['data']['name']} with Dr.{res['data']['doctor']} on {res['data']['date']} at {res['data']['time']}. "
#                 }
#             )
#         else:
#             await p.result_callback(str(res))
#     except Exception as e:
#         print(f"Cancel failed: {str(e)}")
#         await p.result_callback(f"Cancel failed: {str(e)}")
#     finally:
#         llm.set_audio_input_paused(False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


appAPI = FastAPI(lifespan=lifespan)
appAPI.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
appAPI.mount("/static", StaticFiles(directory="static"), name='static')


@appAPI.get('/')
async def index():
    with open("templates/index.html") as f:
        return HTMLResponse(f.read())


@appAPI.websocket('/audio')
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    await run_hosbot(websocket)


async def run_hosbot(websocket: WebSocket):
    with _laminar_session_span("websocket_session"):
        await _run_hosbot_inner(websocket)


async def _run_hosbot_inner(websocket: WebSocket):
    global task, greeted, context_aggregator, llm
    greeted = False
    session = BookingSession()
    tool_active = False
    trace_session_id = f"session-{uuid.uuid4().hex}"
    _laminar_set_trace_session_id(trace_session_id)
    _laminar_set_trace_metadata({
        "transport": "websocket",
    })

    def ns(step: str, extra: dict = {}) -> dict:
        return {"ns": step, **extra}

    # ── INTENT ──────────────────────────────────────────────────────────────
    async def set_intent_call(p):
        nonlocal tool_active
        tool_active = True
        intent = p.arguments.get("intent")
        session.flow = intent
        print(f"[SESSION] flow={intent}")
        _laminar_set_trace_metadata({"flow": intent})
        await p.result_callback(ns(session.next_step()))
        tool_active = False

    # ── APPOINTMENT HISTORY ──────────────────────────────────────────────────
    async def appointment_fetch_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            phone = p.arguments['phone']
            session.patient.phone = phone
            res = await asyncio.to_thread(appointment_fetch, phone)
            print(f"[FETCH] {res}")

            if isinstance(res, list) and res:
                doc = res[0].get('doctor', '')
                await p.result_callback(ns(
                    f"ret_patient:ask if same Dr.{doc} or different. then:{session.next_step()}",
                    {"ret": True, "doc": doc}
                ))
            else:
                await p.result_callback(ns(session.next_step(), {"ret": False}))
        except Exception as e:
            await p.result_callback(ns(session.next_step(), {"err": str(e)}))
        finally:
            tool_active = False

    # ── AVAILABLE SLOTS ──────────────────────────────────────────────────────
    async def available_slots_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            a = p.arguments
            raw_date = a['date']
            for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                try:
                    raw_date = datetime.strptime(
                        raw_date, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    pass

            session.patient.doctor = a['doctor']
            session.patient.date = raw_date

            res = await asyncio.to_thread(available_slots_fun, a['doctor'], raw_date)

            if isinstance(res, list) and res:
                def fmt(s):
                    h, m = s.split(":")
                    hour = int(h)
                    return f"{hour % 12 or 12}:{m} {'AM' if hour < 12 else 'PM'}"

                await gen_ui("slots", {
                    "doctor": a["doctor"],
                    "date": raw_date,
                    "slots": res
                })

                voice_slots = ", ".join(fmt(s) for s in res[:3])
                total = len(res)
                await p.result_callback(
                    f"SLOTS_AVAILABLE: Say exactly this: \"Available slots are {voice_slots}"
                    + (f". More slots are shown on your screen.\" " if total > 3 else ".\" ")
                    + "Then wait for the patient to pick one. Do NOT list any more slots."
                )
            elif isinstance(res, list) and not res:
                await p.result_callback(ns("no_slots:ask different date."))
            else:
                await p.result_callback(ns("slots_err:ask different date."))
        except Exception as e:
            await p.result_callback(ns("slots_err:ask different date."))
        finally:
            tool_active = False

    # ── SHOW CONFIRMATION ────────────────────────────────────────────────────
    # async def confirm_booking_call(p):
    #     llm.set_audio_input_paused(True)
    #     try:
    #         a = p.arguments
    #         if a.get("name"):
    #             session.patient.name = a["name"]
    #         if a.get("age"):
    #             session.patient.age = a["age"]
    #         if a.get("gender"):
    #             session.patient.gender = a["gender"]
    #         if a.get("phone"):
    #             session.patient.phone = a["phone"]
    #         if a.get("doctor"):
    #             session.patient.doctor = a["doctor"]
    #         if a.get("date"):
    #             session.patient.date = a["date"]
    #         if a.get("time"):
    #             session.patient.slot = a["time"]
    #         if a.get("reason"):
    #             session.patient.reason = a["reason"]
    #         if a.get("appointment_id"):
    #             session.patient.appointment_id = a["appointment_id"]

    #         session.awaiting_confirmation = True
    #         is_update = bool(session.patient.appointment_id)
    #         next_action = "waiting_user_yes_to_update" if is_update else "waiting_user_yes_to_book"

    #         await p.result_callback(ns(next_action))
    #     except Exception as e:
    #         await p.result_callback(ns(session.next_step()))
    #     finally:
    #         llm.set_audio_input_paused(False)

    # async def confirm_booking_call(p):
    #     llm.set_audio_input_paused(True)  # Safety
    #     try:
    #         a = p.arguments
    #         # Update session details (Name, Age, Doctor, etc.)
    #         if a.get("name"):
    #             session.patient.name = a["name"]
    #         if a.get("age"):
    #             session.patient.age = a["age"]
    #         if a.get("gender"):
    #             session.patient.gender = a["gender"]
    #         if a.get("phone"):
    #             session.patient.phone = a["phone"]
    #         if a.get("doctor"):
    #             session.patient.doctor = a["doctor"]
    #         if a.get("date"):
    #             session.patient.date = a["date"]
    #         if a.get("time"):
    #             session.patient.slot = a["time"]
    #         if a.get("reason"):
    #             session.patient.reason = a["reason"]

    #         session.awaiting_confirmation = True

    #         # Use the UI helper to show the card on screen
    #         await gen_ui("confirm", a)

    #         # CRITICAL: Return a simple STRING, not a dict
    #         # This prevents the 1007 protocol error
    #         await p.result_callback("CONFIRMATION_SHOWN: Details are on screen. Ask the patient if they are correct.")
    #     except Exception as e:
    #         await p.result_callback("ERROR: Could not show confirmation.")
    #     finally:
    #         await asyncio.sleep(0.1)
    #         llm.set_audio_input_paused(False)

    async def confirm_booking_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            a = p.arguments
            session.patient.name = a.get("name")
            session.patient.age = a.get("age")
            session.patient.gender = a.get("gender")
            session.patient.phone = a.get("phone")
            session.patient.doctor = a.get("doctor")
            session.patient.date = a.get("date")
            session.patient.slot = a.get("time")
            session.patient.reason = a.get("reason")

            session.awaiting_confirmation = True

            await gen_ui("confirm", a)

            await p.result_callback("CONFIRMATION_SHOWN: Ask the patient if these details are correct.")
        except Exception:
            await p.result_callback("ERROR: Unable to process confirmation.")
        finally:
            tool_active = False

    # ── BOOK ─────────────────────────────────────────────────────────────────
    # async def book_appointment_call(p):
    #     llm.set_audio_input_paused(True)
    #     try:
    #         if not session.awaiting_confirmation:
    #             await p.result_callback(ns("call_show_confirmation"))
    #             return

    #         a = p.arguments
    #         res = await asyncio.to_thread(
    #             book_appointment,
    #             a["name"], a['age'], a['gender'], a['phone'],
    #             a['doctor'], a['date'], a['time'], a['reason']
    #         )
    #         if isinstance(res, dict) and res['status'] == 'success':
    #             await gen_ui("summary", res['data'])
    #             session.reset()
    #             await p.result_callback({"ns": "done"})
    #         else:
    #             await p.result_callback(ns("err:small issue. ask retry.", {"e": str(res)[:60]}))
    #     except Exception as e:
    #         await p.result_callback(ns("err:booking failed."))
    #     finally:
    #         llm.set_audio_input_paused(False)

    # async def book_appointment_call(p):
    #     llm.set_audio_input_paused(True)  # Safety guard
    #     try:
    #         if not session.awaiting_confirmation:
    #             await p.result_callback(ns("call_show_confirmation"))
    #             return

    #         a = p.arguments
    #         res = await asyncio.to_thread(
    #             book_appointment,
    #             a["name"], a['age'], a['gender'], a['phone'],
    #             a['doctor'], a['date'], a['time'], a['reason']
    #         )

    #         if isinstance(res, dict) and res.get('status') == 'success':
    #             # Send the complex data to your Frontend UI
    #             await gen_ui("summary", res['data'])
    #             session.reset()

    #             # CRITICAL: Return a SIMPLE flat dictionary to the LLM
    #             # This prevents the 1007 protocol error
    #             await p.result_callback(f"booking confirmed , status: success")
    #             await p.result_callback({"error": "could not save to database", "ns": "retry"})
    #     except Exception as e:
    #         await p.result_callback({"error": "system error during booking"})
    #     finally:
    #         # Small sleep ensures the result frame is fully sent before VAD resumes
    #         await asyncio.sleep(0.1)
    #         llm.set_audio_input_paused(False)

    # async def book_appointment_call(p):
    #     llm.set_audio_input_paused(True)  # Protect the session
    #     try:
    #         if not session.awaiting_confirmation:
    #             await p.result_callback("Error: Confirmation card must be shown first.")
    #             return

    #         a = p.arguments
    #         # Run DB logic in thread
    #         res = await asyncio.to_thread(
    #             book_appointment,
    #             a["name"], a.get('age'), a.get('gender'), a.get('phone'),
    #             a.get('doctor'), a.get('date'), a.get('time'), a.get('reason')
    #         )
    #         print(f"[DB RESULT] {res}")

    #         # Handle SUCCESS (If DB returns a dict with status success)
    #         if isinstance(res, dict) and res.get('status') == 'success':
    #             await gen_ui("summary", res.get('data', {}))
    #             session.reset()
    #             await p.result_callback("BOOKING_SUCCESS: The appointment is confirmed.")

    #         # Handle SUCCESS (If DB returns a string containing 'success')
    #         elif isinstance(res, str) and "status: success" in res:
    #             session.reset()
    #             await p.result_callback("BOOKING_SUCCESS: Your appointment is booked.")

    #         # Handle FAILURE
    #         else:
    #             error_msg = str(res)[:100]
    #             await p.result_callback(f"BOOKING_FAILED: {error_msg}")

    #     except Exception as e:
    #         print(f"Booking callback crash: {e}")
    #         await p.result_callback("SYSTEM_ERROR: Could not complete booking.")
    #     finally:
    #         # Give Gemini time to ingest the result before unpausing mic
    #         await asyncio.sleep(0.2)
    #         llm.set_audio_input_paused(False)

    async def book_appointment_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            a = p.arguments
            res = await asyncio.to_thread(
                book_appointment,
                a["name"], a['age'], a['gender'], a['phone'],
                a['doctor'], a['date'], a['time'], a['reason']
            )

            if isinstance(res, dict) and res.get('status') == 'success':
                d = res['data']
                await gen_ui("summary", d)
                await p.result_callback(
                    f"BOOKING_SUCCESS: Booked for {d['name']} with Dr.{d['doctor']} "
                    f"on {d['date']} at {d['time']}. ID:{d['appointment_id']}."
                )
                session.reset()
            else:
                await p.result_callback(f"BOOKING_FAILED: {str(res)}")

        except Exception as e:
            print(f"[book_appointment_call] exception: {e}")
            await p.result_callback("SYSTEM_ERROR: Booking could not be completed.")
        finally:
            tool_active = False

    # ── FETCH UPCOMING ───────────────────────────────────────────────────────
    async def fetch_upcoming_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            phone = p.arguments['phone']
            session.patient.phone = phone
            res = await asyncio.to_thread(fetch_upcoming_appointments, phone)
            session.upcoming_shown = True

            await gen_ui("upcoming", {'appointments': res, 'phone': phone})

            if res:
                compact = ",".join(
                    f"{a['id']}:{a['doctor'].split('.')[-1]}" for a in res)
                await p.result_callback(ns(session.next_step(), {"apts": compact}))
            else:
                session.reset()
                await p.result_callback(ns("done:no upcoming appointments."))
        except Exception as e:
            await p.result_callback(ns(session.next_step()))
        finally:
            tool_active = False

    # ── UPDATE ───────────────────────────────────────────────────────────────
    async def update_appointment_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            if not session.awaiting_confirmation:
                await p.result_callback(ns(
                    "call_show_confirmation",
                    {"err": "show_confirmation must be called first"}
                ))
                return

            a = p.arguments
            res = await asyncio.to_thread(
                update_appointment,
                a['phone'], a['appointment_id'],
                **{k: v for k, v in a.items() if k not in ['phone', 'appointment_id']}
            )
            if isinstance(res, dict) and res['status'] == 'success':
                d = res['data']
                session.reset()
                await p.result_callback(ns(
                    "done:tell patient appointment updated. ask if anything else.",
                    {"s": "updated", "dr": d['doctor'],
                        "dt": d['date'], "t": d['time']}
                ))
            else:
                await p.result_callback(ns(
                    "err:tell patient small issue updating. ask retry.",
                    {"err": str(res)}
                ))
        except Exception as e:
            await p.result_callback(ns("err:update failed. ask retry.", {"err": str(e)}))
        finally:
            tool_active = False

    # ── CANCEL ───────────────────────────────────────────────────────────────
    async def cancel_appointment_call(p):
        nonlocal tool_active
        tool_active = True
        try:
            a = p.arguments
            confirmed = bool(a.get('confirmed', False))

            if not confirmed:
                details, err = await asyncio.to_thread(
                    get_appointment_for_cancel, a['phone'], a['appointment_id']
                )
                if err:
                    await p.result_callback(ns(f"err:{err}"))
                else:
                    session.cancel_previewed = True
                    session.patient.appointment_id = a['appointment_id']
                    await p.result_callback(ns(
                        "waiting_user_yes_to_cancel",
                        {"apt": f"Dr.{details['doctor']} {details['date']} {details['time']}"}
                    ))
                return

            if not session.cancel_previewed:
                await p.result_callback(ns(
                    "call_cancel_confirmed_false",
                    {"err": "must preview first"}
                ))
                return

            res = await asyncio.to_thread(cancel_appointment, a['phone'], a['appointment_id'])
            if isinstance(res, dict) and res.get('status') == 'success':
                await gen_ui("cancel_summary", res['data'])
                d = res['data']
                session.reset()
                await p.result_callback(ns(
                    "done:tell patient appointment cancelled. ask if anything else.",
                    {"s": "cancelled", "dr": d['doctor'], "dt": d['date']}
                ))
            else:
                await p.result_callback(ns(
                    "err:tell patient issue cancelling. ask retry.",
                    {"err": str(res)}
                ))
        except Exception as e:
            await p.result_callback(ns("err:cancel failed.", {"err": str(e)}))
        finally:
            tool_active = False

    # ── Transport + LLM setup ────────────────────────────────────────────────
    print("client connected via WebSocket")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            serializer=RawPCMSerializer(sample_rate=16000),
        )
    )

    print("llm part")
    llm = GeminiLiveLLMService(
        api_key="",
        model="gemini-3.1-flash-live-preview",
        system_instruction=system_instruction,
        tools=tools,
        inference_on_context_initialization=True,
        voice_id="Aoede",
        params=InputParams(
            modalities=GeminiModalities.AUDIO,
            vad=GeminiVADParams(silence_duration_ms=600),
            thinking=GoogleThinkingConfig(thinking_budget=0),
            context_window_compression=ContextWindowCompressionParams(
                enabled=True,
                trigger_tokens=6000,  # Raised: compression at 2500 was triggering too early
            ),
        ),
        http_options={"api_version": "v1beta"}
    )

    # ── Register all functions ───────────────────────────────────────────────
    llm.register_function("set_intent", set_intent_call)
    llm.register_function("appointment_fetch", appointment_fetch_call)
    llm.register_function("available_slots", available_slots_call)
    llm.register_function("show_confirmation", confirm_booking_call)
    llm.register_function("book_appointment", book_appointment_call)
    llm.register_function("fetch_upcoming_appointments", fetch_upcoming_call)
    llm.register_function("update_appointment", update_appointment_call)
    llm.register_function("cancel_appointment", cancel_appointment_call)

    # ── Context + summarization ──────────────────────────────────────────────
    context = LLMContext(messages=[])
    context_aggregator = LLMContextAggregatorPair(
        context,
        assistant_params=LLMAssistantAggregatorParams(
            enable_auto_context_summarization=False,
            auto_context_summarization_config=LLMAutoContextSummarizationConfig(
                max_context_tokens=800,
                max_unsummarized_messages=3,
                summary_config=LLMContextSummaryConfig(
                    target_context_tokens=400,
                    min_messages_after_summary=2,
                    llm=OpenAILLMService(
                        api_key="f40f6d096d8445fcb644961bdfaa64ce.4jTKHDrfvMQ00giesUjAdFlR",
                        base_url="http://localhost:11434/v1",
                        model="gemma4:31b-cloud"
                    )
                ),
            ),
        ),
    )

    metrics_accumulator = SessionMetricsAccumulator(
        context=context,
        system_instruction=system_instruction,
        api_key="AIzaSyBKNfvwh6im3gMwRioK8R7o1NJnmApWrxA"
    )

    # ── Event handlers ───────────────────────────────────────────────────────
    # @context_aggregator.user().event_handler("on_user_turn_stopped")
    # async def on_userturn_PATCHED(processor, strategy, message):
    #     import re
    #     text = (message.content or "")
    #     print(f"user: {text}")

    #     if not session.patient.phone:
    #         phone_match = re.search(r'\b\d{10}\b', text.replace(" ", ""))
    #         if phone_match:
    #             session.patient.phone = phone_match.group()
    #             print(f"[SESSION] phone captured: {session.patient.phone}")

    #     if session.flow:
    #         snap = session.state_snapshot_if_changed()
    #         if snap is not None:
    #             await task.queue_frames([
    #                 InputTextRawFrame(
    #                     text=f"[S]{json.dumps(snap, separators=(',', ':'))}"

    #                 )
    #             ])
    #             print(f"[STATE injected] {snap}")
    #         else:
    #             print(f"[STATE skipped — unchanged]")
    # state_str = f"\n[STATE]{json.dumps(snap, separators=(',', ':'))}"
    # if hasattr(message, 'content') and message.content:
    #     message.content = message.content + state_str
    # else:
    #     message.content = state_str
    #     print(f"[STATE appended to user message] {snap}")

    # @context_aggregator.user().event_handler("on_user_turn_stopped")
    # async def on_userturn_PATCHED(processor, strategy, message):
    #     text = (message.content or "").strip()
    #     if not text:
    #         return
    #     digits = re.sub(r'\D', '', text)
    #     if len(digits) == 10:
    #         session.patient.phone = digits
    #     snap = session.state_snapshot_if_changed()
    #     if snap and not digits:
    #         await asyncio.sleep(0.1)
    #         state_text = f"[SYSTEM: Current State {json.dumps(snap, separators=(',', ':'))}]"
    #         await task.queue_frames([InputTextRawFrame(text=state_text)])
    #         print(f"[STATE Injected] {snap}")
    #     else:
    #         print(f"[STATE HELD] Tool call (phone detection) in progress.")

    @context_aggregator.user().event_handler("on_user_turn_stopped")
    async def on_userturn_PATCHED(processor, strategy, message):
        text = (message.content or "").strip().lower()
        if not text:
            return

        digits = re.sub(r'\D', '', text)
        if len(digits) == 10:
            session.patient.phone = digits

        if tool_active:
            print(f"[STATE HELD] tool_active=True, skipping injection.")
            return

        snap = session.state_snapshot_if_changed()

        non_digit_content = re.sub(r'\d', '', text).strip()
        pure_digits_only = (len(digits) >= 10 and len(non_digit_content) < 5)
        if snap and not pure_digits_only:
            state_text = f"[SYSTEM: Current State {json.dumps(snap, separators=(',', ':'))}]"
            await task.queue_frames([InputTextRawFrame(text=state_text)])
            print(f"[STATE injected] {snap}")
        elif snap:
            print(f"[STATE HELD] pure digits turn, phone captured silently.")
        else:
            print(f"[STATE HELD] no session change.")

    @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn(processor, message):
        print(f"assistant: {message.content}")
        metrics_accumulator.mark_bot_stopped()

    @context_aggregator.assistant().event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(*args, **kwargs):
        metrics_accumulator.mark_bot_started()

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print("pipeline created")
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,
        LaminarMetricsBridge(),
        metrics_accumulator,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        enable_tracing=True,
        enable_turn_tracking=True,
        conversation_id=trace_session_id,
        observers=[MetricsLogObserver()]
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        print("Queueing initial greeting frame to LLM")
        await task.queue_frames([LLMRunFrame()])
        print("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("Client disconnected")

    runner = PipelineRunner(handle_sigint=False)
    print("pipeline is running")
    await runner.run(task)


@appAPI.get("/greet")
async def greet():
    global greeted
    if task is None:
        return "task not ready"
    if greeted:
        return "already greeted"
    try:
        await task.queue_frames([TTSSpeakFrame(text="Vanakkam, naan Priya, City Clinic-la irunthu pesuren. Ungaluku enna help venum?")])
        print("Greeting queued to LLM")
    except Exception as e:
        print(f"Error occurred while greeting: {e}")
        return "Error occurred while greeting"
    # try:
    #     await task.queue_frames([LLMRunFrame()])
    #     # await task.queue_frames([TTSSpeakFrame(text="Vanakkam, naan Priya, City Clinic-la irunthu pesuren. Ungaluku enna help venum?")])
    #     greeted = True
    #     return "greeted"
    # except Exception as e:
    #     greeted = False
    #     print(f"Error occurred while greeting: {e}")
    #     return "Error occurred while greeting"


@appAPI.get("/stop")
async def stop():
    global greeted
    greeted = False
    return "stopped"


@appAPI.get("/restart")
async def restart():
    global greeted, task
    greeted = False
    if task:
        await task.cancel()
        task = None
    return "restarted"


@appAPI.api_route("/resume", methods=["POST", "GET"])
async def resume(req: Request):
    global task
    msg = {}
    if req.method == "POST":
        try:
            msg = await req.json()
        except Exception:
            msg = {}
    context = msg.get("context", "")
    if not task:
        return "task not ready"
    try:
        prompt = f"""
System: Connection dropped and restored
previous conversation context: {context}
continue naturally. Say "Sorry, connection problem achu — நீங்க என்ன சொல்லீங்க?" and continue from where you left off. Don't re-ask details already given.
"""
        await task.queue_frames([InputTextRawFrame(text=prompt)])
        return "resumed"
    except Exception as e:
        print(f"Error occurred while resuming: {e}")
        return "Error occurred while resuming"


# @appAPI.get("/transcript")
# async def transcript():
#     queue = asyncio.Queue()
#     transcript_client.append(queue)

#     async def generator():
#         try:
#             while True:
#                 msg = await queue.get()
#                 yield f"data: {json.dumps(msg)}\n\n"
#         except asyncio.CancelledError:
#             print("transcript client disconnected")
#             transcript_client.remove(queue)

#     return StreamingResponse(generator(), media_type="text/event-stream")


@appAPI.get("/gen-ui")
async def gen_ui_end():
    queue = asyncio.Queue()
    generat_ui.append(queue)

    async def generator():
        try:
            while True:
                msg = await queue.get()
                yield f"data: {json.dumps(msg)}\n\n"
        except asyncio.CancelledError:
            print("gen ui client disconnected")
            generat_ui.remove(queue)
    return StreamingResponse(generator(), media_type="text/event-stream")


async def gen_ui(event_type: str, data: dict):
    for cl in generat_ui:
        await cl.put({
            "type": event_type,
            "data": data
        })


@appAPI.post("/gen-ui-test")
async def gen_ui_test(req: Request):
    global task
    msg = await req.json()
    txt = msg.get("text", "")
    print(f"gen ui test received: {txt}")
    if not task:
        return "task not ready"
    try:
        await task.queue_frames([
            InputTextRawFrame(text=txt)
        ])
        print(f"injected to LLM: {txt}")
        return "ok"
    except Exception as e:
        print(f"genui err: {str(e)}")
        return f"error: {str(e)}"


# @appAPI.post("/gen-ui-test")
# async def gen_ui_test(req: Request):
#     global task, context_aggregator
#     msg = await req.json()
#     txt = msg.get("text", "")
#     print(f"inject received: {txt}")
#     if not task or not context_aggregator:
#         return "task not ready"
#     try:
#         from pipecat.frames.frames import TextFrame
#         await task.queue_frames([
#             TextFrame(text=txt)
#         ])
#         return "ok"
#     except Exception as e:
#         print(f"inject error: {str(e)}")
#         return f"error: {str(e)}"


#         return "ok"
#     except Exception as e:
#         print(f"genui err: {str(e)}")
#         return "Error occurred while generating UI"


async def send_transcript(role: str, text: str):
    if not text or text.strip() == "" or "<ctrl" in text or "<noise>" in text:
        # await client.put({"role": role, "text": "Please wait for moment"})
        return
    for client in transcript_client:
        await client.put({"role": role, "text": text})


if __name__ == "__main__":
    uvicorn.run(appAPI, host="0.0.0.0", port=5000)

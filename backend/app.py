import asyncio
import os
import sys

import pyaudio
from dotenv import load_dotenv
from loguru import logger
from flask import Flask
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.mistral.llm import MistralLLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyFailover
from pipecat.workers.runner import WorkerRunner
from models import db, Patient, Doctor, Appointment
from pipecat.frames.frames import LLMRunFrame



load_dotenv()

# ── Logger ────────────────────────────────────────────────────────────────────
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # Deepgram Flux requires 16 kHz linear16
CHANNELS      = 1



# app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/hospital'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)

# def book_appointment(name, age, gender, phone, doctor, date=None, boo_k_time=None, reason=None):
#     with app.app_context():
#         try:
#             if date:
#                 for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
#                     try:
#                         date = datetime.strptime(
#                             date, fmt).strftime("%Y-%m-%d")
#                         break
#                     except ValueError:
#                         pass
#             print("booking appointment called")
#             if not boo_k_time:
#                 return f"get the time for booking appointment"
#             if not reason:
#                 return f"get the reason for booking appointment"
#             if not date:
#                 return f"get the date for booking appointment"
#             book_date = datetime.strptime(date, "%Y-%m-%d").date()

#             for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
#                 try:
#                     book_time = datetime.strptime(
#                         boo_k_time.strip(), fmt).time()
#                     break
#                 except ValueError:
#                     pass
#             else:
#                 return f"Invalid time format: {boo_k_time}. Please provide time like 10:45 AM or 14:30"

#             boo_k_time = book_time.strftime("%H:%M")
#             clean_doctor = doctor.replace("Dr.", "").replace("Dr ", "").strip()
#             doctor_name = Doctor.query.filter(
#                 Doctor.name.ilike(f"%{clean_doctor}%")).first()
#             print(
#                 f"clean doctor name: {clean_doctor}, doctor found: {doctor_name.name if doctor_name else 'None'}")
#             print("time :", boo_k_time)
#             phone_exist = Patient.query.filter_by(phone=phone).first()

#             if phone_exist and phone_exist.name != name:
#                 print("phone number exist")
#                 return f"Phone number {phone} is already exist for another patient, please provide a different phone number"

#             if not doctor_name:
#                 return f"Doctor not found: {doctor}"

#             if len(phone) < 10 or len(phone) > 10:
#                 return f"Phone number {phone} is not valid and ask the user once again tell your phone number "

#             if book_date < datetime.now().date():
#                 return f"Date {book_date} is not valid the date has already passed"

#             if doctor_name.work_brk_st_time <= book_time <= doctor_name.work_brk_end_time:
#                 return f"Doctor is on break during {boo_k_time}"

#             if not (doctor_name.work_st_time <= book_time <= doctor_name.work_end_time):
#                 return f"Doctor is not available at {boo_k_time}"

#             already_booked = Appointment.query.filter_by(
#                 doctor_id=doctor_name.id, date=book_date, time=book_time).first()

#             if already_booked:
#                 return f"Doctor is already booked at {boo_k_time} on {date}"

#             patient = Patient.query.filter_by(phone=phone).first()

#             if patient:
#                 existing = Appointment.query.filter_by(
#                     patient_id=patient.id,
#                     doctor_id=doctor_name.id,
#                     date=book_date,
#                     time=book_time
#                 ).first()
#                 if existing:
#                     return f"You have already booked an appointment with Dr. {doctor} on {date} at {boo_k_time}"

#             if not patient:
#                 patient = Patient(
#                     name=name,
#                     age=age,
#                     gender=gender,
#                     phone=phone
#                 )
#                 db.session.add(patient)
#                 db.session.commit()

#             new_apt = Appointment(
#                 patient_id=patient.id,
#                 doctor_id=doctor_name.id,
#                 date=book_date,
#                 time=book_time,
#                 reason=reason
#             )
#             db.session.add(new_apt)
#             db.session.commit()

#             print("booking confirmed")
#             return {
#                 "status": "success",
#                 "data": {
#                     "name": name,
#                     "doctor": doctor_name.name,
#                     "date": date,
#                     "time": boo_k_time,
#                     "reason": reason,
#                     "appointment_id": new_apt.id,
#                 }
#             }
#         except Exception as e:
#             print(f"booking appoint err: str{e}")
#             db.session.rollback()
#             return f"Error{str(e)}"
        



async def main():
    # ── PyAudio instance ───────────────────────────────────────────────────────
    pa = pyaudio.PyAudio()

    # ── Transport ──────────────────────────────────────────────────────────────
    transport = LocalAudioTransport(
    LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=SAMPLE_RATE,
        audio_out_sample_rate=SAMPLE_RATE,
    )
)
    
    stt = DeepgramFluxSTTService(
    api_key="ac390c1a6291805ab580c7f4050c225e34deb7cb",
    settings=DeepgramFluxSTTService.Settings(
        model="flux-general-en",
        eot_threshold=0.7,
        keyterm=["your", "terms"],
    ),
)

    # # ── LLM : Gemini 2.5 Flash ────────────────────────────────────────────────
    # llm = GoogleLLMService(
    #     api_key="AIzaSyD5-0t4hyRbPFaQrudY-cdahWQ-IbW8ilg",
    #     model="gemini-2.5-flash",
    #     # Pipecat auto-disables thinking for gemini-2.5-flash to cut latency.
    #     # Pass settings=GoogleLLMService.Settings(system_instruction=...) if
    #     # you need more fine-grained control.
    # )

    SYSTEM_INSTRUCTION = "You are a helpful assistant in a voice conversation. Keep responses like humans with emotions tone and the text response you are give was sent to the tts for audio response so give the response that should mimic like real humans."

    llm_gemini = GoogleLLMService(
    api_key="AIzaSyD5-0t4hyRbPFaQrudY-cdahWQ-IbW8ilg",
    settings=GoogleLLMService.Settings(
        model="gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
        max_tokens=1024,
    ),
)

    # Fallback 1: Mistral
    llm_mistral = MistralLLMService(
        api_key="sics84YZ5sbBCPmXQhnfmZzro3L7qOUm",
        settings=MistralLLMService.Settings(
            model="mistral-medium-latest",
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_tokens=1024,
        ),
    )

    # Fallback 2: OpenRouter
    llm_openrouter = OpenRouterLLMService(
        api_key="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",
        settings=OpenRouterLLMService.Settings(
            model="openrouter/auto",
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_tokens=1024,
        ),
    )


    tts = CartesiaTTSService(
        api_key="sk_car_GFSZXscziXzjJHEH3bBhzz",
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
        settings=CartesiaTTSService.Settings(
            model="sonic-2",
        ),
    )


    llm_switcher = LLMSwitcher(
    llms=[llm_gemini, llm_mistral, llm_openrouter],
    strategy_type=ServiceSwitcherStrategyFailover
)

    # ── Conversation context ───────────────────────────────────────────────────
    context = LLMContext( 
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
        ]
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context,
                                                                     user_params=LLMUserAggregatorParams(
                                                                         vad_analyzer=SileroVADAnalyzer(
                                                                             params=VADParams(stop_secs=0.2)
                                                                         ),
                                                                     ),)

    pipeline = Pipeline(
        [
            transport.input(),               # PyAudio mic frames
            stt,                             # Deepgram Flux → TranscriptionFrame
            user_aggregator,                 # accumulate user text into context
            llm_switcher,                    # LLM switcher → LLM text frames
            tts,                             # Cartesia → audio frames
            transport.output(),              # play through speakers
            assistant_aggregator,            # accumulate assistant reply into context
        ]
    )

    # ── Task ───────────────────────────────────────────────────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,    # barge-in handled by Silero VAD
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message("developer", "Say hello and briefly introduce yourself.")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    # ── Run ────────────────────────────────────────────────────────────────────
    logger.info("🎤  Voice agent ready — speak into your mic. Ctrl+C to quit.")
    runner = WorkerRunner()
    await runner.add_workers(task)
    await runner.run()

    pa.terminate()


if __name__ == "__main__":
    asyncio.run(main())
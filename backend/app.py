import asyncio
import os
import sys
from contextlib import asynccontextmanager
import re
from datetime import datetime, time
from models import SessionLocal, Patient, Doctor, Appointment, Base, engine
import uvicorn
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame, InputTextRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
# from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import InputAudioRawFrame, AudioRawFrame as _AudioRawFrame
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.mistral.llm import MistralLLMService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyFailover
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
import os
from pipecat.workers.runner import WorkerRunner
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
# from pipecat.services.google import GoogleTTSService
# from pipecat.transcriptions.language import Language
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up to stt-llm-tts/
load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")  
logger.add(
    "logs/app.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,
    mode="a",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
)
logger.level("SESSION", no=25, color="<green>", icon="🟢")

def log_session_separator():
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log("SESSION", "\n\n" + "=" * 60)
    logger.log("SESSION", f"NEW SESSION STARTED — {session_id}")
    logger.log("SESSION", "=" * 60)


AUDIO_IN_SAMPLE_RATE  = 16_000   
AUDIO_OUT_SAMPLE_RATE = 24_000   

task: PipelineWorker | None = None
greeted: bool = False


class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                # STT, LLM, TTS — all emit TTFB
                if isinstance(d, TTFBMetricsData):
                    logger.info(f"[{d.processor}] TTFB: {d.value}s")

                # STT, LLM, TTS — all emit Processing Time
                elif isinstance(d, ProcessingMetricsData):
                    logger.info(f"[{d.processor}] Processing Time: {d.value}s")

                # LLM only
                elif isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    logger.info(
                        f"[{d.processor}] LLM Tokens — "
                        f"prompt: {tokens.prompt_tokens}, "
                        f"completion: {tokens.completion_tokens}, "
                        f"total: {tokens.total_tokens}"
                    )

                # TTS only
                elif isinstance(d, TTSUsageMetricsData):
                    logger.info(f"[{d.processor}] TTS Characters: {d.value}")

                # TTS sentence aggregation latency
                elif isinstance(d, TextAggregationMetricsData):
                    logger.info(f"[{d.processor}] Text Aggregation: {d.value}s")

        await self.push_frame(frame, direction)




class RawPCMSerializer(FrameSerializer):
  
    def __init__(self, sample_rate: int = AUDIO_IN_SAMPLE_RATE):
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


# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from models import DATABASE_URL
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    try:
        parent_url, db_name = DATABASE_URL.rsplit('/', 1)
        parent_url = parent_url + '/'
        
        logger.info(f"Checking if database '{db_name}' exists...")
        temp_engine = create_async_engine(parent_url, isolation_level="AUTOCOMMIT")
        async with temp_engine.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        await temp_engine.dispose()
    except Exception as e:
        logger.warning(f"Could not check/create database programmatically: {e}")

    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully!")
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend/static")), name="static")


#---Function Calling---

# async def book_appointment(name, age, gender, phone, doctor, date=None, boo_k_time=None, reason=None):
#     async with SessionLocal() as session:
#         try:
#             if date:
#                 for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
#                     try:
#                         date = datetime.strptime(date, fmt).strftime("%Y-%m-%d")
#                         break
#                     except ValueError:
#                         pass
#             print("booking appointment called")
#             if not boo_k_time:
#                 return "get the time for booking appointment"
#             if not reason:
#                 return "get the reason for booking appointment"
#             if not date:
#                 return "get the date for booking appointment"
#             book_date = datetime.strptime(date, "%Y-%m-%d").date()

#             for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
#                 try:
#                     book_time = datetime.strptime(boo_k_time.strip(), fmt).time()
#                     break
#                 except ValueError:
#                     pass
#             else:
#                 return f"Invalid time format: {boo_k_time}. Please provide time like 10:45 AM or 14:30"

#             boo_k_time = book_time.strftime("%H:%M")
#             clean_doctor = doctor.replace("Dr.", "").replace("Dr ", "").strip()
            
#             stmt_doc = select(Doctor).filter(Doctor.name.ilike(f"%{clean_doctor}%"))
#             res_doc = await session.execute(stmt_doc)
#             doctor_obj = res_doc.scalars().first()
            
#             print(f"clean doctor name: {clean_doctor}, doctor found: {doctor_obj.name if doctor_obj else 'None'}")
#             print("time :", boo_k_time)
            
#             stmt_p_exist = select(Patient).filter_by(phone=phone)
#             res_p_exist = await session.execute(stmt_p_exist)
#             phone_exist = res_p_exist.scalars().first()

#             if phone_exist and phone_exist.name != name:
#                 print("phone number exist")
#                 return f"Phone number {phone} is already exist for another patient, please provide a different phone number"

#             if not doctor_obj:
#                 return f"Doctor not found: {doctor}"

#             if len(phone) < 10 or len(phone) > 10:
#                 return f"Phone number {phone} is not valid and ask the user once again tell your phone number "

#             if book_date < datetime.now().date():
#                 return f"Date {book_date} is not valid the date has already passed"

#             if doctor_obj.work_brk_st_time <= book_time <= doctor_obj.work_brk_end_time:
#                 return f"Doctor is on break during {boo_k_time}"

#             if not (doctor_obj.work_st_time <= book_time <= doctor_obj.work_end_time):
#                 return f"Doctor is not available at {boo_k_time}"

#             stmt_booked = select(Appointment).filter_by(doctor_id=doctor_obj.id, date=book_date, time=book_time)
#             res_booked = await session.execute(stmt_booked)
#             already_booked = res_booked.scalars().first()

#             if already_booked:
#                 return f"Doctor is already booked at {boo_k_time} on {date}"

#             stmt_pat = select(Patient).filter_by(phone=phone)
#             res_pat = await session.execute(stmt_pat)
#             patient = res_pat.scalars().first()

#             if patient:
#                 stmt_existing = select(Appointment).filter_by(
#                     patient_id=patient.id,
#                     doctor_id=doctor_obj.id,
#                     date=book_date,
#                     time=book_time
#                 )
#                 res_existing = await session.execute(stmt_existing)
#                 existing = res_existing.scalars().first()
#                 if existing:
#                     return f"You have already booked an appointment with Dr. {doctor} on {date} at {boo_k_time}"

#             if not patient:
#                 patient = Patient(
#                     name=name,
#                     age=age,
#                     gender=gender,
#                     phone=phone
#                 )
#                 session.add(patient)
#                 await session.commit()
#                 await session.refresh(patient)

#             new_apt = Appointment(
#                 patient_id=patient.id,
#                 doctor_id=doctor_obj.id,
#                 date=book_date,
#                 time=book_time,
#                 reason=reason
#             )
#             session.add(new_apt)
#             await session.commit()
#             await session.refresh(new_apt)

#             print("booking confirmed")
#             return {
#                 "status": "success",
#                 "data": {
#                     "name": name,
#                     "doctor": doctor_obj.name,
#                     "date": date,
#                     "time": boo_k_time,
#                     "reason": reason,
#                     "appointment_id": new_apt.id,
#                 }
#             }
#         except Exception as e:
#             print(f"booking appoint err: str{e}")
#             await session.rollback()
#             return f"Error {str(e)}"

async def book_appointment(name, age, gender, phone, doctor, date=None, boo_k_time=None, reason=None):
    async with SessionLocal() as session:
        try:
            if date:
                for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                    try:
                        date = datetime.strptime(date, fmt).strftime("%Y-%m-%d")
                        break
                    except ValueError:
                        pass

            if not boo_k_time:
                return "get the time for booking appointment"
            if not reason:
                return "get the reason for booking appointment"
            if not date:
                return "get the date for booking appointment"

            book_date = datetime.strptime(date, "%Y-%m-%d").date()

            for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                try:
                    book_time = datetime.strptime(boo_k_time.strip(), fmt).time()
                    break
                except ValueError:
                    pass
            else:
                return f"Invalid time format: {boo_k_time}."

            boo_k_time = book_time.strftime("%H:%M")
            clean_doctor = doctor.replace("Dr.", "").replace("Dr ", "").strip()

            stmt_doc = select(Doctor).filter(Doctor.name.ilike(f"%{clean_doctor}%"))
            res_doc = await session.execute(stmt_doc)
            doctor_obj = res_doc.scalars().first()

            if not doctor_obj:
                return f"Doctor not found: {doctor}"

            # ✅ Capture name BEFORE any commit/refresh
            doctor_name = doctor_obj.name
            doctor_id = doctor_obj.id

            if len(phone) < 10 or len(phone) > 10:
                return f"Phone number {phone} is not valid"

            if book_date < datetime.now().date():
                return f"Date {book_date} has already passed"

            if doctor_obj.work_brk_st_time <= book_time <= doctor_obj.work_brk_end_time:
                return f"Doctor is on break during {boo_k_time}"

            if not (doctor_obj.work_st_time <= book_time <= doctor_obj.work_end_time):
                return f"Doctor is not available at {boo_k_time}"

            stmt_booked = select(Appointment).filter_by(
                doctor_id=doctor_id, date=book_date, time=book_time
            )
            res_booked = await session.execute(stmt_booked)
            if res_booked.scalars().first():
                return f"Doctor is already booked at {boo_k_time} on {date}"

            stmt_p_exist = select(Patient).filter_by(phone=phone)
            res_p_exist = await session.execute(stmt_p_exist)
            phone_exist = res_p_exist.scalars().first()

            if phone_exist and phone_exist.name != name:
                return f"Phone number {phone} already exists for another patient"

            stmt_pat = select(Patient).filter_by(phone=phone)
            res_pat = await session.execute(stmt_pat)
            patient = res_pat.scalars().first()

            if patient:
                stmt_existing = select(Appointment).filter_by(
                    patient_id=patient.id,
                    doctor_id=doctor_id,
                    date=book_date,
                    time=book_time
                )
                res_existing = await session.execute(stmt_existing)
                if res_existing.scalars().first():
                    return f"You already have an appointment with Dr. {doctor_name} on {date} at {boo_k_time}"

            if not patient:
                patient = Patient(name=name, age=age, gender=gender, phone=phone)
                session.add(patient)
                await session.flush()  # ✅ Use flush() to get patient.id without committing
                # DON'T refresh here — flush is enough to populate patient.id

            patient_id = patient.id  # ✅ Capture before commit

            new_apt = Appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                date=book_date,
                time=book_time,
                reason=reason
            )
            session.add(new_apt)
            await session.flush()  # ✅ flush to get new_apt.id

            apt_id = new_apt.id  # ✅ Capture before commit

            await session.commit()  # ✅ Now commit — no more ORM access after this

            return {
                "status": "success",
                "data": {
                    "name": name,
                    "doctor": doctor_name,   # ✅ Already captured above
                    "date": date,
                    "time": boo_k_time,
                    "reason": reason,
                    "appointment_id": apt_id,  # ✅ Already captured above
                }
            }

        except Exception as e:
            print(f"booking appoint err: {str(e)}")
            await session.rollback()
            return f"Error {str(e)}"

async def available_slots_fun(doctor: str, date: str):
    async with SessionLocal() as session:
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

            stmt_doc = select(Doctor).filter(Doctor.name.ilike(f"%{doctor}%"))
            res_doc = await session.execute(stmt_doc)
            doc = res_doc.scalars().first()
            if not doc:
                normalized = normalize_name(doctor)
                if normalized:
                    res_all = await session.execute(select(Doctor))
                    for candidate in res_all.scalars().all():
                        if normalize_name(candidate.name) == normalized:
                            doc = candidate
                            break
            print(f"doctor name: {doctor}")
            if not doc:
                print(f"No doctor found with name: {doctor}")
                return f"No doctor found with name: {doctor}"
                
            stmt_booked = select(Appointment).filter_by(doctor_id=doc.id, date=date)
            res_booked = await session.execute(stmt_booked)
            booked = res_booked.scalars().all()
            
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
            print(f"start:{start},end:{end},brk_start:{brk_start},brk_end:{brk_end}")
            today = datetime.now().date()
            date_v = datetime.strptime(date, "%Y-%m-%d").date()
            for i in range(start, end, doc.meet_duration):
                hour = i // 60
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
            return f"Error {str(e)}"


async def appointment_fetch(phone):
    async with SessionLocal() as session:
        try:
            if len(phone) != 10:
                print(f"Invalid phone number length: {phone}")
                return f"Invalid phone number length: {phone}"
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return f"No patient found with phone number: {phone}"
                
            stmt_a = select(Appointment).filter_by(patient_id=patient.id)
            res_a = await session.execute(stmt_a)
            appoint = res_a.scalars().all()
            if not appoint:
                return f"No previous appointments found for phone number: {phone}"
            apt_list = []
            for apt in appoint:
                stmt_d = select(Doctor).filter_by(id=apt.doctor_id)
                res_d = await session.execute(stmt_d)
                doctor = res_d.scalars().first()
                print(f"Appointment with Dr. {doctor.name} on {apt.date} at {apt.time}, Reason: {apt.reason}, Patient name:{patient.name}")
                apt_list.append({
                    "doctor": doctor.name,
                    "date": str(apt.date),
                    "time": str(apt.time)[:5],
                    "reason": apt.reason
                })
            return apt_list
        except Exception as e:
            print(f"error:{str(e)}")
            return f"appointment fetch details Error {str(e)}"


async def update_appointment(phone, appointment_id, **updates):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return f"No patient found with phone number: {phone}"

            stmt_a = select(Appointment).filter_by(
                id=appointment_id,
                patient_id=patient.id,
            )
            res_a = await session.execute(stmt_a)
            apt = res_a.scalars().first()
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
                    target_date = datetime.strptime(updates['date'], "%Y-%m-%d").date()
                except ValueError:
                    return f"Invalid date format: {updates['date']}. Please provide date like YYYY-MM-DD"

            if 'time' in updates and updates['time']:
                parsed_time = None
                for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                    try:
                        parsed_time = datetime.strptime(updates['time'].strip(), fmt).time()
                        break
                    except ValueError:
                        pass
                if not parsed_time:
                    return f"Invalid time format: {updates['time']}. Please provide time like 10:45 AM or 14:30"
                target_time = parsed_time

            if 'reason' in updates and updates['reason']:
                apt.reason = updates['reason']

            if 'doctor' in updates and updates['doctor']:
                stmt_d = select(Doctor).filter(
                    Doctor.name.ilike(f"%{updates['doctor']}%")
                )
                res_d = await session.execute(stmt_d)
                doc = res_d.scalars().first()
                if not doc:
                    return f"Doctor not found: {updates['doctor']}"
                target_doctor_id = doc.id

            target_dt = datetime.combine(target_date, target_time)
            if target_dt < now_dt:
                return "Cannot update to a passed date/time. Previous booking date and time are kept unchanged."

            stmt_td = select(Doctor).filter_by(id=target_doctor_id)
            res_td = await session.execute(stmt_td)
            target_doctor = res_td.scalars().first()
            if not target_doctor:
                return "Doctor details not found for update"

            if target_doctor.work_brk_st_time <= target_time <= target_doctor.work_brk_end_time:
                return f"Doctor is on break during {target_time.strftime('%H:%M')}. Previous booking date and time are kept unchanged."

            if not (target_doctor.work_st_time <= target_time <= target_doctor.work_end_time):
                return f"Doctor is not available at {target_time.strftime('%H:%M')}. Previous booking date and time are kept unchanged."

            stmt_ab = select(Appointment).filter(
                Appointment.doctor_id == target_doctor_id,
                Appointment.date == target_date,
                Appointment.time == target_time,
                Appointment.id != apt.id,
            )
            res_ab = await session.execute(stmt_ab)
            already_booked = res_ab.scalars().first()
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

            await session.commit()
            await session.refresh(apt)
            await session.refresh(patient)
            
            stmt_fd = select(Doctor).filter_by(id=apt.doctor_id)
            res_fd = await session.execute(stmt_fd)
            doctor = res_fd.scalars().first()
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
            await session.rollback()
            print(f"update appointment err: {str(e)}")
            return f"Error: {str(e)}"


async def fetch_upcoming_appointments(phone):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return []
            stmt_a = select(Appointment).filter_by(patient_id=patient.id)
            res_a = await session.execute(stmt_a)
            apts = res_a.scalars().all()
            upcoming = []
            today = datetime.now().date()
            for apt in apts:
                if apt.date >= today:
                    stmt_d = select(Doctor).filter_by(id=apt.doctor_id)
                    res_d = await session.execute(stmt_d)
                    doctor = res_d.scalars().first()
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


async def cancel_appointment(phone, appointment_id):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return f"No patient found with phone number: {phone}"

            stmt_a = select(Appointment).filter_by(
                id=appointment_id,
                patient_id=patient.id,
            )
            res_a = await session.execute(stmt_a)
            apt = res_a.scalars().first()
            if not apt:
                return f"No appointment found with id: {appointment_id}"

            apt_dt = datetime.combine(apt.date, apt.time)
            if apt_dt < datetime.now():
                return "Cannot cancel past appointment"

            stmt_d = select(Doctor).filter_by(id=apt.doctor_id)
            res_d = await session.execute(stmt_d)
            doctor = res_d.scalars().first()
            cancelled_data = {
                'appointment_id': apt.id,
                'name': patient.name,
                'phone': patient.phone,
                'doctor': doctor.name if doctor else "Unknown",
                'date': str(apt.date),
                'time': str(apt.time)[:5],
                'reason': apt.reason,
            }

            await session.delete(apt)
            await session.commit()

            return {
                'status': 'success',
                'message': 'Appointment cancelled successfully',
                'data': cancelled_data,
            }
        except Exception as e:
            await session.rollback()
            print(f"cancel appointment err: {str(e)}")
            return f"Error: {str(e)}"


async def get_appointment_for_cancel(phone, appointment_id):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return None, f"No patient found with phone number: {phone}"

            stmt_a = select(Appointment).filter_by(
                id=appointment_id,
                patient_id=patient.id,
            )
            res_a = await session.execute(stmt_a)
            apt = res_a.scalars().first()
            if not apt:
                return None, f"No appointment found with id: {appointment_id}"

            apt_dt = datetime.combine(apt.date, apt.time)
            if apt_dt < datetime.now():
                return None, "Cannot cancel past appointment"

            stmt_d = select(Doctor).filter_by(id=apt.doctor_id)
            res_d = await session.execute(stmt_d)
            doctor = res_d.scalars().first()
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
            pass
    return f"Invalid time format: {time_str}. Please provide time like 10:45 AM or 14:30"


#-----Function Schemas-----

book_appointment_schema = FunctionSchema(
    name="book_appointment",
    description="Save the medical appointment details to the database once confirmed.",
    properties={
        "name": {
            "type": "string",
            "description": "The full name of the patient.",
        },
        "age": {
            "type": "integer",
            "description": "The age of the patient in years.",
        },
        "gender": {
            "type": "string",
            "description": "The gender of the patient.",
        },
        "phone": {
            "type": "string",
            "description": "The 10-digit phone number of the patient.",
        },
        "doctor": {
            "type": "string",
            "description": "The name of the doctor (e.g., Dr. Shakthi).",
        },
        "date": {
            "type": "string",
            "description": "The date of the appointment in YYYY-MM-DD format.",
        },
        "time": {
            "type": "string",
            "description": "The time of the appointment in HH:MM format (24-hour format, e.g., 14:30 or 10:00).",
        },
        "reason": {
            "type": "string",
            "description": "The reason for booking the appointment.",
        },
    },
    required=["name", "age", "gender", "phone", "doctor", "date", "time", "reason"]
)

available_slots_schema = FunctionSchema(
    name="available_slots",
    description="Fetch available open time slots for a specific doctor on a given date.",
    properties={
        "doctor": {
            "type": "string",
            "description": "The name of the doctor.",
        },
        "date": {
            "type": "string",
            "description": "The date to check for slots in YYYY-MM-DD format.",
        },
    },
    required=["doctor", "date"]
)

appointment_fetch_schema = FunctionSchema(
    name="appointment_fetch",
    description="Fetch patient's last appointment history by phone number.",
    properties={
        "phone": {
            "type": "string",
            "description": "The 10-digit phone number of the patient.",
        }
    },
    required=["phone"]
)

fetch_upcoming_appointments_schema = FunctionSchema(
    name="fetch_upcoming_appointments",
    description="Fetch upcoming appointments associated with a phone number for update or cancel flow.",
    properties={
        "phone": {
            "type": "string",
            "description": "The patient's 10-digit phone number.",
        }
    },
    required=["phone"]
)

update_appointment_schema = FunctionSchema(
    name="update_appointment",
    description="Update an existing appointment in the database.",
    properties={
        "phone": {
            "type": "string",
            "description": "The patient's 10-digit phone number.",
        },
        "appointment_id": {
            "type": "integer",
            "description": "The ID of the appointment to update.",
        },
        "name": {
            "type": "string",
            "description": "The new name of the patient (optional).",
        },
        "age": {
            "type": "integer",
            "description": "The new age of the patient (optional).",
        },
        "gender": {
            "type": "string",
            "description": "The new gender of the patient (optional).",
        },
        "doctor": {
            "type": "string",
            "description": "The new doctor name (optional).",
        },
        "date": {
            "type": "string",
            "description": "The new appointment date (YYYY-MM-DD) (optional).",
        },
        "time": {
            "type": "string",
            "description": "The new appointment time (HH:MM format) (optional).",
        },
        "reason": {
            "type": "string",
            "description": "The new reason for the appointment (optional).",
        },
    },
    required=["phone", "appointment_id"]
)

cancel_appointment_schema = FunctionSchema(
    name="cancel_appointment",
    description="Cancel appointment. Set confirmed=false to preview/find it first, or confirmed=true to execute the cancellation.",
    properties={
        "phone": {
            "type": "string",
            "description": "The patient's 10-digit phone number.",
        },
        "appointment_id": {
            "type": "integer",
            "description": "The ID of the appointment to cancel.",
        },
        "confirmed": {
            "type": "boolean",
            "description": "Flag to confirm the cancellation. False to preview, True to delete.",
        },
    },
    required=["phone", "appointment_id"]
)



@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(BASE_DIR, "frontend/templates/index.html")) as f:
        return f.read()


@app.websocket("/audio")
async def audio_ws(websocket: WebSocket):
    await websocket.accept()
    await run_pipeline(websocket)


async def run_pipeline(websocket: WebSocket):
    global task, greeted

    greeted = False

    log_session_separator()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
            serializer=RawPCMSerializer(sample_rate=AUDIO_IN_SAMPLE_RATE),
            # audio_in_filter=RNNoiseFilter(),

        ),
    )

    stt = DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
        settings=DeepgramFluxSTTService.Settings(
            model="flux-general-en",
            eot_threshold=0.7,
            keyterm=[],
        ),
    )

    # stt = DeepgramSTTService(
    #     api_key="",
    #     settings=DeepgramSTTService.Settings(
    #         model="nova-3-general",
    #     ),
    # )



    current_date = datetime.now().strftime("%A, %d %B %Y")  
    current_time = datetime.now().strftime("%I:%M %p")  

    SYSTEM_INSTRUCTION = f"""
    Role:
    You are Priya, a warm and friendly human receptionist at a hospital. You handle appointment bookings, doctor information, and patient enquiries for City Clinic.

    Persona:
        Speak like a real hospital receptionist — warm, quick, empathetic.
        You are NOT an AI. Never break character.
        Respond like a busy but kind hospital receptionist — 
        professional, focused, and efficient. 
        Do NOT compliment patient names, ages, or personal details.
        Do NOT say things like "what a lovely name", "nice name", 
        "that's great", or any hollow filler praise.
        Just acknowledge and move to the next question naturally.

    Language:
        English

    ---

    RESPONSE FORMAT — CRITICAL FOR TTS:

    Your responses will be spoken aloud via a text-to-speech system (Gpt 4o mini tts).

    ---

    CONVERSATION RULES:

    - All responses must be under 30 words.
    - Collect these details for booking (one at a time, naturally):
        Name → Age → Gender → Phone number → Reason → Date → Time (slot selection)
    - Ask one detail at a time, like a real human would.
    - Never give technical database errors to patients. Say "small internal issue" instead.
    - Convert all 24h times to 12h AM/PM before speaking (e.g., 14:30 → 2:30 PM).
    - If all slots are available, say "all slots are open" — don't list individual times.

    - When patient gives personal details (name, age, gender), 
    just acknowledge briefly and move on naturally.
    Keep it warm but task-focused — like a receptionist 
    who genuinely cares but has other patients to attend to.

    WRONG: "Oh what a lovely name! Now may I know your age?"
    WRONG: "Got it." (too cold/robotic)
    RIGHT: "Alright [name], and how old are you?"
    RIGHT: "Okay, and what seems to be the issue today?"

    DATE & TIME HANDLING:

        Current date awareness:
            - Today's date is: {current_date}
            - Current time is: {current_time}
            - Use these to resolve all relative date references.
            - Resolve relative dates internally before any tool call:
                "today"        → current date
                "tomorrow"     → current date + 1
                "day after"    → current date + 2
                "this Monday"  → nearest upcoming Monday
                "next Monday"  → Monday of the following week
                "next week"    → 7 days from today
            - NEVER ask patient to say date in YYYY-MM-DD format.
            Convert silently, confirm naturally.

        Always confirm resolved date back to patient:
            WRONG: silently pass date to tool
            RIGHT: "So that's this Thursday, the 19th — is that right?"
            
            If patient says a weekday that already passed this week:
            → Confirm: "Did you mean this coming Monday, or next Monday?"
            
            If patient says "Monday" ambiguously:
            → Ask: "Just to confirm — did you mean Monday the 23rd 
                    or Monday the 30th?"

        Time handling:
            - If patient says "morning" → ask: "Any preference between 
            9 AM to 12 PM, or shall I check all morning slots?"
            - If patient says "afternoon" → 12 PM to 5 PM range
            - If patient says "evening" → 5 PM onwards
            - Always speak time in 12h AM/PM format, never 24h.

    ---

    STT ERROR GUARDRAILS:

        If transcribed text sounds garbled or doesn't make sense 
        in context, don't guess — ask again naturally:
            "Sorry, I didn't catch that clearly. Could you say it again?"

        Common STT misrecognitions to watch for:
            - Reason field: "figure" → probably "fever"
                            "my grain" → probably "migraine"  
                            "pressure" → probably "blood pressure"
            - Numbers: "to" / "too" → could be "2"
                    "for" → could be "4"
            - If a number field (age, phone) has non-numeric 
            words, ask again: "Could you repeat that number 
            for me slowly?"

        Phone number guardrails:
            - Must be exactly 10 digits.
            - If patient gives less or more, say:
            "I think I missed a digit — could you repeat 
            your number slowly?"
            - Never pass an incomplete number to any tool.

    ---

    GENERAL GUARDRAILS:

        Incomplete info after 2 tries:
            - Move on and come back: "No worries, we can 
            come back to that. What's your phone number?"

        Never:
            - Reveal tool names, function names, or database errors
            - Ask more than one question at a time
            - Confirm a booking without repeating all details back
            - Pass garbled, incomplete, or unresolved data to any tool

    ---

    TOOL / FUNCTION CALLING (CURRENT IMPLEMENTATION):

    Use these exact tools:
        1. `appointment_fetch`           → fetch patient's last appointment history by phone.
        2. `available_slots`             → check open time slots for a doctor on a specific date.
        3. `book_appointment`            → save appointment to the database once details are confirmed.
        4. `fetch_upcoming_appointments` → fetch upcoming appointments associated with a phone number (used for update/cancel).
        5. `update_appointment`          → update an existing appointment.
        6. `cancel_appointment`          → cancel an existing appointment. Set confirmed=false first to preview, and confirmed=true to execute.

    When searching or booking by doctor name:
        - Strip "Dr." prefix before passing name to functions (e.g., pass "Shakthi" or "Ramya").
        - Always add "Dr." back when speaking to the patient (e.g., say "Dr. Shakthi").

    ---

    WORKFLOW:

    Booking Flow:
        1. Collect Name, Age, Gender, Reason.
        2. Assign Doctor:
        - If age <= 15: Assign Dr. Ramya (Pediatrician).
        - If age > 15: Assign Dr. Shakthi (General Physician).
        If patient requests Dr. Shakthi but is <= 15, or has pediatric reason, suggest Dr. Ramya instead.
        3. Check Slots:
        Query `available_slots` for the assigned doctor and date. 
        Present 3 slots verbatim. If slots are open, say "all slots are open". If booked, suggest a different time.
        4. Ask Phone:
        Collect phone number. Query `appointment_fetch` to see if they are a returning patient.
        If a duplicate phone database error returns, ask the user to provide a different phone number.
        5. Show Confirmation:
        Summarize details, then call `book_appointment` after user says yes.

    Update Flow:
        1. Ask for Phone.
        2. Call `fetch_upcoming_appointments` to find their bookings.
        3. Ask which appointment to update and what details to change.
        4. Call `update_appointment` once details are confirmed.

    Cancel Flow:
        1. Ask for Phone.
        2. Call `fetch_upcoming_appointments` to find their bookings.
        3. Ask which appointment to cancel.
        4. Call `cancel_appointment` with confirmed=false to preview it.
        5. Confirm with patient, then call `cancel_appointment` with confirmed=true to delete.
"""



    # llm_gemini = GoogleLLMService(
    #     =os.getenv("GOOGLE_API_KEY", ""),
    #     settings=GoogleLLMService.Settings(
    #         model="gemini-2.5-flash-lite",
    #         system_instruction=SYSTEM_INSTRUCTION,
    #         temperature=0.7,
    #         max_tokens=1024,
    #     ),
    # )

    llm_mistral = MistralLLMService(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
        settings=MistralLLMService.Settings(
            model="mistral-medium-latest",
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_tokens=1024,
        ),
    )

    llm_openrouter = OpenRouterLLMService(
        api_key=os.getenv(
            "OPENROUTER_API_KEY",
            "",
        ),
        settings=OpenRouterLLMService.Settings(
            model="openrouter/auto",
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_tokens=1024,
        ),
    )

    llm_switcher = LLMSwitcher(
        llms=[ llm_mistral, llm_openrouter],
        strategy_type=ServiceSwitcherStrategyFailover,
    )

    async def book_appointment_call(p: FunctionCallParams):
        try:
            a = p.arguments
            res = await book_appointment(
                name=a.get("name"),
                age=a.get("age"),
                gender=a.get("gender"),
                phone=a.get("phone"),
                doctor=a.get("doctor"),
                date=a.get("date"),
                boo_k_time=a.get("time"),
                reason=a.get("reason")
            )
            if isinstance(res, dict) and res.get('status') == 'success':
                d = res['data']
                await p.result_callback(
                    f"BOOKING_SUCCESS: Booked for {d['name']} with Dr.{d['doctor']} "
                    f"on {d['date']} at {d['time']}. ID:{d['appointment_id']}."
                )
            else:
                await p.result_callback(f"BOOKING_FAILED: {str(res)}")
        except Exception as e:
            await p.result_callback(f"SYSTEM_ERROR: Booking could not be completed: {str(e)}")

    async def available_slots_call(p: FunctionCallParams):
        try:
            a = p.arguments
            res = await available_slots_fun(doctor=a.get("doctor"), date=a.get("date"))
            if isinstance(res, list):
                if res:
                    voice_slots = ", ".join(res[:3])
                    await p.result_callback(f"SLOTS_AVAILABLE: {voice_slots}. Tell these slots to the patient.")
                else:
                    await p.result_callback("no_slots: No slots available on this date.")
            else:
                await p.result_callback(f"slots_err: {str(res)}")
        except Exception as e:
            await p.result_callback(f"slots_err: Error fetching slots: {str(e)}")

    async def appointment_fetch_call(p: FunctionCallParams):
        try:
            phone = p.arguments.get("phone")
            res = await appointment_fetch(phone)
            if isinstance(res, list) and res:
                doc = res[0].get('doctor', '')
                await p.result_callback(f"patient_history: Last appointment was with Dr. {doc} on {res[0].get('date')} at {res[0].get('time')}.")
            else:
                await p.result_callback("no_history: No previous appointments found.")
        except Exception as e:
            await p.result_callback(f"history_err: {str(e)}")

    async def fetch_upcoming_appointments_call(p: FunctionCallParams):
        try:
            phone = p.arguments.get("phone")
            res = await fetch_upcoming_appointments(phone)
            if res:
                compact = ", ".join(f"ID {a['id']} with Dr. {a['doctor']} on {a['date']} at {a['time']}" for a in res)
                await p.result_callback(f"upcoming_appointments: {compact}")
            else:
                await p.result_callback("no_upcoming: No upcoming appointments found.")
        except Exception as e:
            await p.result_callback(f"upcoming_err: {str(e)}")

    async def update_appointment_call(p: FunctionCallParams):
        try:
            a = p.arguments
            phone = a.get("phone")
            appointment_id = a.get("appointment_id")
            updates = {k: v for k, v in a.items() if k not in ["phone", "appointment_id"]}
            res = await update_appointment(phone, appointment_id, **updates)
            if isinstance(res, dict) and res.get('status') == 'success':
                d = res['data']
                await p.result_callback(f"UPDATE_SUCCESS: Appointment updated. New details: Dr. {d['doctor']} on {d['date']} at {d['time']}.")
            else:
                await p.result_callback(f"UPDATE_FAILED: {str(res)}")
        except Exception as e:
            await p.result_callback(f"SYSTEM_ERROR: Update failed: {str(e)}")

    async def cancel_appointment_call(p: FunctionCallParams):
        try:
            a = p.arguments
            phone = a.get("phone")
            appointment_id = a.get("appointment_id")
            confirmed = a.get("confirmed", False)
            if not confirmed:
                details, err = await get_appointment_for_cancel(phone, appointment_id)
                if err:
                    await p.result_callback(f"err: {err}")
                else:
                    await p.result_callback(f"preview_cancel: Found appointment with Dr. {details['doctor']} on {details['date']} at {details['time']}. Ask the user if they are sure they want to cancel.")
            else:
                res = await cancel_appointment(phone, appointment_id)
                if isinstance(res, dict) and res.get('status') == 'success':
                    await p.result_callback("CANCEL_SUCCESS: Appointment cancelled successfully.")
                else:
                    await p.result_callback(f"CANCEL_FAILED: {str(res)}")
        except Exception as e:
            await p.result_callback(f"SYSTEM_ERROR: Cancellation failed: {str(e)}")

    llm_switcher.register_function("book_appointment", book_appointment_call)
    llm_switcher.register_function("available_slots", available_slots_call)
    llm_switcher.register_function("appointment_fetch", appointment_fetch_call)
    llm_switcher.register_function("fetch_upcoming_appointments", fetch_upcoming_appointments_call)
    llm_switcher.register_function("update_appointment", update_appointment_call)
    llm_switcher.register_function("cancel_appointment", cancel_appointment_call)


    tools =ToolsSchema(standard_tools=[book_appointment_schema, available_slots_schema, appointment_fetch_schema, fetch_upcoming_appointments_schema, update_appointment_schema, cancel_appointment_schema])

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
        settings=CartesiaTTSService.Settings(
            model="sonic-3.5",
        ),
    )


    # tts = OpenAITTSService(
    #     api_key="",
    #     settings=OpenAITTSService.Settings(
    #         model="gpt-4o-mini-tts",
    #         voice="nova", 
    #         instructions=f"""
    #         Voice Identity: Consistently maintain a low tone, friendly female voice throughout the entire conversation. Do not change gender, pitch profile, or vocal identity between responses.
    #         Voice Style: low tone, soft, empathetic, and professional, reassuring the customer that their issue is understood and will be resolved.
    #         Punctuation: Well-structured with natural pauses, allowing for clarity and a steady, calming flow.
    #         Delivery: Calm and patient, with a supportive and understanding tone that reassures the listener.
    #         Phrasing: Clear and concise, using customer-friendly language that avoids jargon while maintaining professionalism.
    #         Tone: Empathetic and solution-focused, emphasizing both understanding and proactive assistance.
    #         """ ,
    #         speed=1.15,
    #     ),
    # )

    context = LLMContext()

    context.set_tools(tools)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    # start_secs=0.2,
                    # stop_secs=0.2,
                    # min_volume=0.6,
                    start_secs=0.3,
                    stop_secs=0.2,  
                    confidence=0.7,
                    min_volume=0.6,
                    )
            ),
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            user_turn_stop_timeout=2.0,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),        
            stt,                     
            user_aggregator,          
            llm_switcher,            
            tts,   
            assistant_aggregator,                  
            transport.output(),    
            MetricsLogger(),   
        ]
    )

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,   
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[MetricsLogObserver()],

    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("WebSocket client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("WebSocket client disconnected")
        await task.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    await runner.run()




@app.get("/greet")
async def greet():
    global greeted, task
    if task is None:
        return PlainTextResponse("task not ready")
    if greeted:
        return PlainTextResponse("already greeted")
    try:
        await task.queue_frames([
            TTSSpeakFrame(text="Hello, I'm Priya from City Clinic. How can I help you today?")
        ])
        greeted = True
        logger.info("Greeting sent")
        return PlainTextResponse("greeted")
    except Exception as e:
        logger.error(f"Greet error: {e}")
        return PlainTextResponse("error")


@app.get("/restart")
async def restart():
    global greeted, task
    greeted = False
    if task:
        await task.cancel()
        task = None
    logger.info("Pipeline restarted")
    return PlainTextResponse("restarted")


@app.api_route("/resume", methods=["POST", "GET"])
async def resume(req: Request):
    
    global task
    msg = {}
    if req.method == "POST":
        try:
            msg = await req.json()
        except Exception:
            pass

    context_text = msg.get("context", "")
    if not task:
        return PlainTextResponse("task not ready")

    try:
        prompt = (
            "System: Connection dropped and restored.\n"
            f"Previous conversation:\n{context_text}\n"
            "Continue naturally from where we left off. "
            "Briefly acknowledge the reconnect and continue."
        )
        await task.queue_frames([InputTextRawFrame(text=prompt)])
        logger.info("Resume context injected")
        return PlainTextResponse("resumed")
    except Exception as e:
        logger.error(f"Resume error: {e}")
        return PlainTextResponse("error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

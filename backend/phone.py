import asyncio
import os
import sys
import re
from contextlib import asynccontextmanager
from datetime import datetime, time

import uvicorn
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from models import SessionLocal, Patient, Doctor, Appointment, Base, engine

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
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
from pipecat.workers.runner import WorkerRunner
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy

from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.runner.utils import parse_telephony_websocket


load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
logger.add(
    "logs/phone.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,
    mode="a",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
)

# ── Exotel uses 8kHz mono PCM 16-bit linear audio ─────────────────────────────
EXOTEL_SAMPLE_RATE = 8_000


# ── FastAPI lifespan: DB init ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from models import DATABASE_URL
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    try:
        parent_url, db_name = DATABASE_URL.rsplit("/", 1)
        parent_url = parent_url + "/"
        temp_engine = create_async_engine(parent_url, isolation_level="AUTOCOMMIT")
        async with temp_engine.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
        await temp_engine.dispose()
    except Exception as e:
        logger.warning(f"Could not check/create database: {e}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized.")
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB helper functions ────────────────────────────────────────────────────────

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
                    patient_id=patient.id, doctor_id=doctor_id,
                    date=book_date, time=book_time,
                )
                res_existing = await session.execute(stmt_existing)
                if res_existing.scalars().first():
                    return f"You already have an appointment with Dr. {doctor_name} on {date} at {boo_k_time}"

            if not patient:
                patient = Patient(name=name, age=age, gender=gender, phone=phone)
                session.add(patient)
                await session.flush()

            patient_id = patient.id
            new_apt = Appointment(
                patient_id=patient_id, doctor_id=doctor_id,
                date=book_date, time=book_time, reason=reason,
            )
            session.add(new_apt)
            await session.flush()
            apt_id = new_apt.id
            await session.commit()

            return {
                "status": "success",
                "data": {
                    "name": name, "doctor": doctor_name,
                    "date": date, "time": boo_k_time,
                    "reason": reason, "appointment_id": apt_id,
                },
            }
        except Exception as e:
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
            if not doc:
                return f"No doctor found with name: {doctor}"

            stmt_booked = select(Appointment).filter_by(doctor_id=doc.id, date=date)
            res_booked = await session.execute(stmt_booked)
            booked = res_booked.scalars().all()

            avai_slot = []
            current_time = datetime.now().time()
            start = doc.work_st_time.hour * 60 + doc.work_st_time.minute
            end = doc.work_end_time.hour * 60 + doc.work_end_time.minute
            brk_start = doc.work_brk_st_time.hour * 60 + doc.work_brk_st_time.minute
            brk_end = doc.work_brk_end_time.hour * 60 + doc.work_brk_end_time.minute
            booked_slot = [t.time for t in booked]

            today = datetime.now().date()
            date_v = datetime.strptime(date, "%Y-%m-%d").date()
            for i in range(start, end, doc.meet_duration):
                hour = i // 60
                minute = i % 60
                time_slt = time(hour, minute)
                if time_slt in booked_slot:
                    continue
                if brk_start <= i < brk_end:
                    continue
                if today == date_v and time_slt <= current_time:
                    continue
                avai_slot.append(f"{hour:02d}:{minute:02d}")
            return avai_slot
        except Exception as e:
            return f"Error {str(e)}"


async def appointment_fetch(phone):
    async with SessionLocal() as session:
        try:
            if len(phone) != 10:
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
                apt_list.append({
                    "doctor": doctor.name,
                    "date": str(apt.date),
                    "time": str(apt.time)[:5],
                    "reason": apt.reason,
                })
            return apt_list
        except Exception as e:
            return f"Error {str(e)}"


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
                        "id": apt.id, "doctor": doctor.name,
                        "date": str(apt.date), "time": str(apt.time)[:5],
                        "reason": apt.reason, "name": patient.name,
                        "age": patient.age, "gender": patient.gender,
                    })
            return upcoming
        except Exception as e:
            return []


async def update_appointment(phone, appointment_id, **updates):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return f"No patient found with phone number: {phone}"

            stmt_a = select(Appointment).filter_by(id=appointment_id, patient_id=patient.id)
            res_a = await session.execute(stmt_a)
            apt = res_a.scalars().first()
            if not apt:
                return f"No appointment found with id: {appointment_id}"

            now_dt = datetime.now()
            if datetime.combine(apt.date, apt.time) < now_dt:
                return "Cannot update past appointment"

            old_date = apt.date
            old_time = apt.time
            target_date = apt.date
            target_time = apt.time
            target_doctor_id = apt.doctor_id

            if "date" in updates and updates["date"]:
                try:
                    target_date = datetime.strptime(updates["date"], "%Y-%m-%d").date()
                except ValueError:
                    return f"Invalid date format: {updates['date']}"

            if "time" in updates and updates["time"]:
                parsed_time = None
                for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                    try:
                        parsed_time = datetime.strptime(updates["time"].strip(), fmt).time()
                        break
                    except ValueError:
                        pass
                if not parsed_time:
                    return f"Invalid time format: {updates['time']}"
                target_time = parsed_time

            if "reason" in updates and updates["reason"]:
                apt.reason = updates["reason"]

            if "doctor" in updates and updates["doctor"]:
                stmt_d = select(Doctor).filter(Doctor.name.ilike(f"%{updates['doctor']}%"))
                res_d = await session.execute(stmt_d)
                doc = res_d.scalars().first()
                if not doc:
                    return f"Doctor not found: {updates['doctor']}"
                target_doctor_id = doc.id

            if datetime.combine(target_date, target_time) < now_dt:
                return "Cannot update to a passed date/time."

            stmt_td = select(Doctor).filter_by(id=target_doctor_id)
            res_td = await session.execute(stmt_td)
            target_doctor = res_td.scalars().first()
            if not target_doctor:
                return "Doctor details not found"

            if target_doctor.work_brk_st_time <= target_time <= target_doctor.work_brk_end_time:
                return f"Doctor is on break during {target_time.strftime('%H:%M')}."

            if not (target_doctor.work_st_time <= target_time <= target_doctor.work_end_time):
                return f"Doctor not available at {target_time.strftime('%H:%M')}."

            stmt_ab = select(Appointment).filter(
                Appointment.doctor_id == target_doctor_id,
                Appointment.date == target_date,
                Appointment.time == target_time,
                Appointment.id != apt.id,
            )
            res_ab = await session.execute(stmt_ab)
            if res_ab.scalars().first():
                return f"Doctor already booked at {target_time.strftime('%H:%M')} on {target_date}."

            apt.date = target_date
            apt.time = target_time
            apt.doctor_id = target_doctor_id

            if "name" in updates and updates["name"]:
                patient.name = updates["name"]
            if "age" in updates and updates["age"]:
                patient.age = updates["age"]
            if "gender" in updates and updates["gender"]:
                patient.gender = updates["gender"]

            await session.commit()
            await session.refresh(apt)
            await session.refresh(patient)

            stmt_fd = select(Doctor).filter_by(id=apt.doctor_id)
            res_fd = await session.execute(stmt_fd)
            doctor = res_fd.scalars().first()
            return {
                "status": "success",
                "data": {
                    "name": patient.name, "age": patient.age, "gender": patient.gender,
                    "phone": phone, "doctor": doctor.name,
                    "date": str(apt.date), "time": str(apt.time)[:5],
                    "reason": apt.reason, "appointment_id": apt.id,
                    "previous_date": str(old_date), "previous_time": str(old_time)[:5],
                },
            }
        except Exception as e:
            await session.rollback()
            return f"Error: {str(e)}"


async def get_appointment_for_cancel(phone, appointment_id):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return None, f"No patient found with phone number: {phone}"

            stmt_a = select(Appointment).filter_by(id=appointment_id, patient_id=patient.id)
            res_a = await session.execute(stmt_a)
            apt = res_a.scalars().first()
            if not apt:
                return None, f"No appointment found with id: {appointment_id}"

            if datetime.combine(apt.date, apt.time) < datetime.now():
                return None, "Cannot cancel past appointment"

            stmt_d = select(Doctor).filter_by(id=apt.doctor_id)
            res_d = await session.execute(stmt_d)
            doctor = res_d.scalars().first()
            return {
                "appointment_id": apt.id, "name": patient.name, "phone": patient.phone,
                "doctor": doctor.name if doctor else "Unknown",
                "date": str(apt.date), "time": str(apt.time)[:5], "reason": apt.reason,
            }, None
        except Exception as e:
            return None, f"Error: {str(e)}"


async def cancel_appointment(phone, appointment_id):
    async with SessionLocal() as session:
        try:
            stmt_p = select(Patient).filter_by(phone=phone)
            res_p = await session.execute(stmt_p)
            patient = res_p.scalars().first()
            if not patient:
                return f"No patient found with phone number: {phone}"

            stmt_a = select(Appointment).filter_by(id=appointment_id, patient_id=patient.id)
            res_a = await session.execute(stmt_a)
            apt = res_a.scalars().first()
            if not apt:
                return f"No appointment found with id: {appointment_id}"

            if datetime.combine(apt.date, apt.time) < datetime.now():
                return "Cannot cancel past appointment"

            stmt_d = select(Doctor).filter_by(id=apt.doctor_id)
            res_d = await session.execute(stmt_d)
            doctor = res_d.scalars().first()
            cancelled_data = {
                "appointment_id": apt.id, "name": patient.name, "phone": patient.phone,
                "doctor": doctor.name if doctor else "Unknown",
                "date": str(apt.date), "time": str(apt.time)[:5], "reason": apt.reason,
            }
            await session.delete(apt)
            await session.commit()
            return {"status": "success", "message": "Appointment cancelled successfully", "data": cancelled_data}
        except Exception as e:
            await session.rollback()
            return f"Error: {str(e)}"


# ── Function Schemas ───────────────────────────────────────────────────────────

book_appointment_schema = FunctionSchema(
    name="book_appointment",
    description="Save the medical appointment details to the database once confirmed.",
    properties={
        "name":   {"type": "string",  "description": "Full name of the patient."},
        "age":    {"type": "integer", "description": "Age of the patient in years."},
        "gender": {"type": "string",  "description": "Gender of the patient."},
        "phone":  {"type": "string",  "description": "10-digit phone number of the patient."},
        "doctor": {"type": "string",  "description": "Name of the doctor."},
        "date":   {"type": "string",  "description": "Appointment date in YYYY-MM-DD format."},
        "time":   {"type": "string",  "description": "Appointment time in HH:MM format."},
        "reason": {"type": "string",  "description": "Reason for the appointment."},
    },
    required=["name", "age", "gender", "phone", "doctor", "date", "time", "reason"],
)

available_slots_schema = FunctionSchema(
    name="available_slots",
    description="Fetch available open time slots for a specific doctor on a given date.",
    properties={
        "doctor": {"type": "string", "description": "Name of the doctor."},
        "date":   {"type": "string", "description": "Date to check in YYYY-MM-DD format."},
    },
    required=["doctor", "date"],
)

appointment_fetch_schema = FunctionSchema(
    name="appointment_fetch",
    description="Fetch patient's last appointment history by phone number.",
    properties={
        "phone": {"type": "string", "description": "10-digit phone number of the patient."},
    },
    required=["phone"],
)

fetch_upcoming_appointments_schema = FunctionSchema(
    name="fetch_upcoming_appointments",
    description="Fetch upcoming appointments associated with a phone number for update or cancel flow.",
    properties={
        "phone": {"type": "string", "description": "Patient's 10-digit phone number."},
    },
    required=["phone"],
)

update_appointment_schema = FunctionSchema(
    name="update_appointment",
    description="Update an existing appointment in the database.",
    properties={
        "phone":          {"type": "string",  "description": "Patient's 10-digit phone number."},
        "appointment_id": {"type": "integer", "description": "ID of the appointment to update."},
        "name":           {"type": "string",  "description": "New patient name (optional)."},
        "age":            {"type": "integer", "description": "New patient age (optional)."},
        "gender":         {"type": "string",  "description": "New patient gender (optional)."},
        "doctor":         {"type": "string",  "description": "New doctor name (optional)."},
        "date":           {"type": "string",  "description": "New date YYYY-MM-DD (optional)."},
        "time":           {"type": "string",  "description": "New time HH:MM (optional)."},
        "reason":         {"type": "string",  "description": "New reason (optional)."},
    },
    required=["phone", "appointment_id"],
)

cancel_appointment_schema = FunctionSchema(
    name="cancel_appointment",
    description="Cancel appointment. confirmed=false to preview, confirmed=true to execute.",
    properties={
        "phone":          {"type": "string",  "description": "Patient's 10-digit phone number."},
        "appointment_id": {"type": "integer", "description": "ID of the appointment to cancel."},
        "confirmed":      {"type": "boolean", "description": "False to preview, True to delete."},
    },
    required=["phone", "appointment_id"],
)


# ── Exotel HTTP Webhook Handlers ──────────────────────────────────────────────
# Exotel may fire an HTTP GET/POST to the URL before the WebSocket upgrade.
# These handlers acknowledge the request so Exotel proceeds to WebSocket.

from fastapi.responses import PlainTextResponse

@app.get("/")
async def exotel_http_root(request: Request):
    """
    Exotel sends an HTTP GET to the configured URL first (especially with
    Passthru-style flows). Returning 200 OK lets it proceed.
    Log the call parameters for debugging.
    """
    params = dict(request.query_params)
    logger.info(f"Exotel HTTP GET / — params: {params}")
    return PlainTextResponse("OK", status_code=200)

@app.post("/")
async def exotel_http_root_post(request: Request):
    body = await request.body()
    logger.info(f"Exotel HTTP POST / — body: {body}")
    return PlainTextResponse("OK", status_code=200)

@app.get("/ws")
async def exotel_http_ws(request: Request):
    """HTTP pre-flight to /ws — some Exotel flows hit this before WebSocket upgrade."""
    params = dict(request.query_params)
    logger.info(f"Exotel HTTP GET /ws — params: {params}")
    return PlainTextResponse("OK", status_code=200)


# ── Exotel Inbound WebSocket Endpoint ─────────────────────────────────────────

@app.websocket("/ws")
async def exotel_ws(websocket: WebSocket):
    
    await websocket.accept()
    logger.info("Exotel inbound call WebSocket connected")

    try:
        transport_type, call_data = await parse_telephony_websocket(websocket)

        stream_id   = call_data.get("stream_id", "")
        call_id     = call_data.get("call_id", "")
        from_number = call_data.get("from", "")   
        to_number   = call_data.get("to", "")     

        logger.info(f"Call — From: {from_number} | To: {to_number} | Call ID: {call_id}")

        caller_phone = from_number.lstrip("+").lstrip("91") if from_number else ""
        caller_phone = caller_phone[-10:] if len(caller_phone) >= 10 else caller_phone

        await run_phone_pipeline(websocket, stream_id, call_id, caller_phone)

    except Exception as e:
        logger.error(f"Exotel WebSocket error: {e}")


async def run_phone_pipeline(
    websocket: WebSocket,
    stream_id: str,
    call_id: str,
    caller_phone: str,
):
    serializer = ExotelFrameSerializer(
        stream_sid=stream_id,
        call_sid=call_id,
    )

    # ── Transport: same FastAPIWebsocketTransport with Exotel serializer ───────
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=EXOTEL_SAMPLE_RATE,
            audio_out_sample_rate=EXOTEL_SAMPLE_RATE,
            serializer=serializer,
        ),
    )

    # ── STT ────────────────────────────────────────────────────────────────────
    stt = DeepgramFluxSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY", "ac390c1a6291805ab580c7f4050c225e34deb7cb"),
        settings=DeepgramFluxSTTService.Settings(
            model="flux-general-en",
            eot_threshold=0.7,
            keyterm=[],
        ),
    )

    # ── System Prompt ──────────────────────────────────────────────────────────
    current_date = datetime.now().strftime("%A, %d %B %Y")
    current_time_str = datetime.now().strftime("%I:%M %p")

    caller_hint = (
        f"The caller's phone number is {caller_phone}. "
        "Silently call appointment_fetch with this number before greeting. "
        "If found, greet them by name as a returning patient. "
        "Also skip asking for their phone number during booking since you already have it."
        if caller_phone else ""
    )

    SYSTEM_INSTRUCTION = f"""
    Role:
    You are Priya, a warm and friendly human receptionist at a hospital. You handle appointment bookings, doctor information, and patient enquiries for City Clinic.

    Persona:
        Speak like a real hospital receptionist — warm, quick, empathetic.
        You are NOT an AI. Never break character.
        Respond like a busy but kind hospital receptionist — professional, focused, and efficient.
        Do NOT compliment patient names, ages, or personal details.
        Do NOT say things like "what a lovely name", "nice name", "that's great", or any hollow filler praise.
        Just acknowledge and move to the next question naturally.

    Language:
        English

    {caller_hint}

    ---

    RESPONSE FORMAT — CRITICAL FOR TTS:
    Your responses will be spoken aloud via a text-to-speech system.
    All responses must be under 30 words.

    ---

    CONVERSATION RULES:

    - Collect these details for booking (one at a time, naturally):
        Name → Age → Gender → Phone number → Reason → Date → Time (slot selection)
    - Ask one detail at a time, like a real human would.
    - Never give technical database errors to patients. Say "small internal issue" instead.
    - Convert all 24h times to 12h AM/PM before speaking (e.g., 14:30 → 2:30 PM).
    - If all slots are available, say "all slots are open" — don't list individual times.

    DATE & TIME HANDLING:
        Current date awareness:
            - Today's date is: {current_date}
            - Current time is: {current_time_str}
            - Resolve relative dates internally:
                "today"       → current date
                "tomorrow"    → current date + 1
                "day after"   → current date + 2
                "this Monday" → nearest upcoming Monday
                "next Monday" → Monday of the following week
            - NEVER ask patient to say date in YYYY-MM-DD format. Convert silently.
        Always confirm resolved date back to patient.
        Time: Always speak in 12h AM/PM format.

    ---

    STT ERROR GUARDRAILS:
        If transcribed text sounds garbled, ask again:
            "Sorry, I didn't catch that clearly. Could you say it again?"
        Phone number: Must be exactly 10 digits. Never pass incomplete numbers to tools.

    ---

    TOOL / FUNCTION CALLING:

    Use these exact tools:
        1. appointment_fetch           → fetch patient's last appointment history by phone.
        2. available_slots             → check open time slots for a doctor on a specific date.
        3. book_appointment            → save appointment once details are confirmed.
        4. fetch_upcoming_appointments → fetch upcoming appointments (used for update/cancel).
        5. update_appointment          → update an existing appointment.
        6. cancel_appointment          → cancel appointment. confirmed=false to preview, confirmed=true to execute.

    When booking by doctor name: strip "Dr." prefix before passing to functions, add "Dr." back when speaking.

    ---

    WORKFLOW:

    Booking Flow:
        1. Collect Name, Age, Gender, Reason.
        2. Assign Doctor:
           - Age <= 15 → Dr. Ramya (Pediatrician)
           - Age > 15  → Dr. Shakthi (General Physician)
        3. Check Slots: Query available_slots. Present up to 3 slots.
        4. Collect Phone. Query appointment_fetch to check returning patient.
        5. Confirm all details → call book_appointment.

    Update Flow:
        1. Ask for Phone → call fetch_upcoming_appointments.
        2. Ask what to change → call update_appointment.

    Cancel Flow:
        1. Ask for Phone → call fetch_upcoming_appointments.
        2. Call cancel_appointment with confirmed=false to preview.
        3. Confirm with patient → call cancel_appointment with confirmed=true.
    """

    # ── LLMs with failover ─────────────────────────────────────────────────────
    llm_mistral = MistralLLMService(
        api_key=os.getenv("MISTRAL_API_KEY", "sics84YZ5sbBCPmXQhnfmZzro3L7qOUm"),
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
            "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",
        ),
        settings=OpenRouterLLMService.Settings(
            model="openrouter/auto",
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_tokens=1024,
        ),
    )

    llm_switcher = LLMSwitcher(
        llms=[llm_mistral, llm_openrouter],
        strategy_type=ServiceSwitcherStrategyFailover,
    )

    # ── Function call handlers ─────────────────────────────────────────────────

    async def book_appointment_call(p: FunctionCallParams):
        try:
            a = p.arguments
            res = await book_appointment(
                name=a.get("name"), age=a.get("age"), gender=a.get("gender"),
                phone=a.get("phone"), doctor=a.get("doctor"),
                date=a.get("date"), boo_k_time=a.get("time"), reason=a.get("reason"),
            )
            if isinstance(res, dict) and res.get("status") == "success":
                d = res["data"]
                await p.result_callback(
                    f"BOOKING_SUCCESS: Booked for {d['name']} with Dr.{d['doctor']} "
                    f"on {d['date']} at {d['time']}. ID:{d['appointment_id']}."
                )
            else:
                await p.result_callback(f"BOOKING_FAILED: {str(res)}")
        except Exception as e:
            await p.result_callback(f"SYSTEM_ERROR: {str(e)}")

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
            await p.result_callback(f"slots_err: {str(e)}")

    async def appointment_fetch_call(p: FunctionCallParams):
        try:
            phone = p.arguments.get("phone")
            res = await appointment_fetch(phone)
            if isinstance(res, list) and res:
                doc = res[0].get("doctor", "")
                await p.result_callback(
                    f"patient_history: Last appointment was with Dr. {doc} "
                    f"on {res[0].get('date')} at {res[0].get('time')}."
                )
            else:
                await p.result_callback("no_history: No previous appointments found.")
        except Exception as e:
            await p.result_callback(f"history_err: {str(e)}")

    async def fetch_upcoming_appointments_call(p: FunctionCallParams):
        try:
            phone = p.arguments.get("phone")
            res = await fetch_upcoming_appointments(phone)
            if res:
                compact = ", ".join(
                    f"ID {a['id']} with Dr. {a['doctor']} on {a['date']} at {a['time']}" for a in res
                )
                await p.result_callback(f"upcoming_appointments: {compact}")
            else:
                await p.result_callback("no_upcoming: No upcoming appointments found.")
        except Exception as e:
            await p.result_callback(f"upcoming_err: {str(e)}")

    async def update_appointment_call(p: FunctionCallParams):
        try:
            a = p.arguments
            updates = {k: v for k, v in a.items() if k not in ["phone", "appointment_id"]}
            res = await update_appointment(a.get("phone"), a.get("appointment_id"), **updates)
            if isinstance(res, dict) and res.get("status") == "success":
                d = res["data"]
                await p.result_callback(
                    f"UPDATE_SUCCESS: Updated. Dr. {d['doctor']} on {d['date']} at {d['time']}."
                )
            else:
                await p.result_callback(f"UPDATE_FAILED: {str(res)}")
        except Exception as e:
            await p.result_callback(f"SYSTEM_ERROR: {str(e)}")

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
                    await p.result_callback(
                        f"preview_cancel: Found appointment with Dr. {details['doctor']} "
                        f"on {details['date']} at {details['time']}. Ask the user to confirm cancellation."
                    )
            else:
                res = await cancel_appointment(phone, appointment_id)
                if isinstance(res, dict) and res.get("status") == "success":
                    await p.result_callback("CANCEL_SUCCESS: Appointment cancelled successfully.")
                else:
                    await p.result_callback(f"CANCEL_FAILED: {str(res)}")
        except Exception as e:
            await p.result_callback(f"SYSTEM_ERROR: {str(e)}")

    llm_switcher.register_function("book_appointment", book_appointment_call)
    llm_switcher.register_function("available_slots", available_slots_call)
    llm_switcher.register_function("appointment_fetch", appointment_fetch_call)
    llm_switcher.register_function("fetch_upcoming_appointments", fetch_upcoming_appointments_call)
    llm_switcher.register_function("update_appointment", update_appointment_call)
    llm_switcher.register_function("cancel_appointment", cancel_appointment_call)

    tools = ToolsSchema(
        standard_tools=[
            book_appointment_schema,
            available_slots_schema,
            appointment_fetch_schema,
            fetch_upcoming_appointments_schema,
            update_appointment_schema,
            cancel_appointment_schema,
        ]
    )

    # ── TTS (Cartesia at 8kHz for Exotel) ─────────────────────────────────────
    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY", "sk_car_GFSZXscziXzjJHEH3bBhzz"),
    #     voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
    #     settings=CartesiaTTSService.Settings(
    #         model="sonic-3.5",
    #         # sample_rate=EXOTEL_SAMPLE_RATE,
    #     ),
    # )

    tts = OpenAITTSService(
        api_key="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",
        settings=OpenAITTSService.Settings(
            model="gpt-4o-mini-tts",
            voice="nova", 
            instructions=f"""
            Voice Identity: Consistently maintain a low tone, friendly female voice throughout the entire conversation. Do not change gender, pitch profile, or vocal identity between responses.
            Voice Style: low tone, soft, empathetic, and professional, reassuring the customer that their issue is understood and will be resolved.
            Punctuation: Well-structured with natural pauses, allowing for clarity and a steady, calming flow.
            Delivery: Calm and patient, with a supportive and understanding tone that reassures the listener.
            Phrasing: Clear and concise, using customer-friendly language that avoids jargon while maintaining professionalism.
            Tone: Empathetic and solution-focused, emphasizing both understanding and proactive assistance.
            """ ,
            speed=1.15,
        ),
    )

    # ── Context & Aggregators ──────────────────────────────────────────────────
    context = LLMContext()
    context.set_tools(tools)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    start_secs=0.3,
                    stop_secs=0.3,
                    confidence=0.7,
                    min_volume=0.6,
                )
            ),
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            user_turn_stop_timeout=2.0,
        ),
    )

    # ── Pipeline ───────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm_switcher,
            tts,
            assistant_aggregator,
            transport.output(),
        ]
    )

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=EXOTEL_SAMPLE_RATE,
            audio_out_sample_rate=24_000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Exotel audio stream started — sending greeting")
        await task.queue_frames([
            TTSSpeakFrame(
                text="Hello, thank you for calling City Clinic. This is Priya. How can I help you today?"
            )
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Exotel call disconnected")
        await task.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    await runner.run()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)

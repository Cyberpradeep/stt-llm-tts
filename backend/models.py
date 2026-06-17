import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, DateTime, Time, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+aiomysql://root:root@localhost:3306/hospital")

engine = create_async_engine(DATABASE_URL, pool_recycle=3600, pool_pre_ping=True)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()


class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    phone = Column(String(20), nullable=False, unique=True)
    create_time = Column(DateTime, default=datetime.utcnow)

    appointments_t = relationship("Appointment", back_populates="patient")


class Doctor(Base):
    __tablename__ = 'doctors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    specialty = Column(String(100), nullable=False)
    work_st_time = Column(Time, nullable=False)
    work_end_time = Column(Time, nullable=False)
    work_brk_st_time = Column(Time, nullable=False)
    work_brk_end_time = Column(Time, nullable=False)
    days_work = Column(String(100), nullable=False)
    meet_duration = Column(Integer, nullable=False)

    appointments_t = relationship("Appointment", back_populates="doctor")


class Appointment(Base):
    __tablename__ = 'appointments'
    
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    doctor_id = Column(Integer, ForeignKey('doctors.id'), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    reason = Column(String(255), nullable=False)
    create_time = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="appointments_t")
    doctor = relationship("Doctor", back_populates="appointments_t")
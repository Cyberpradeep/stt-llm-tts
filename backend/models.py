from flask_sqlalchemy import SQLAlchemy 
from sqlalchemy.orm import relationship
from datetime import datetime
db = SQLAlchemy()

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(20), nullable=False, unique=True)
    create_time= db.Column(db.DateTime, default=datetime.utcnow)

    appointments_t=relationship("Appointment", back_populates="patient")

class Doctor(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)
    work_st_time= db.Column(db.Time, nullable=False)
    work_end_time= db.Column(db.Time, nullable=False)
    work_brk_st_time= db.Column(db.Time, nullable=False)
    work_brk_end_time= db.Column(db.Time, nullable=False)
    days_work= db.Column(db.String(100), nullable=False)
    meet_duration= db.Column(db.Integer, nullable=False)

    appointments_t=relationship("Appointment", back_populates="doctor")

class Appointment(db.Model):
    __tablename__ = 'appointments'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    date= db.Column(db.Date, nullable=False)
    time= db.Column(db.Time, nullable=False)
    reason= db.Column(db.String(255), nullable=False)
    create_time= db.Column(db.DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="appointments_t")
    doctor = relationship("Doctor", back_populates="appointments_t")
"""
Database models for LearnADo application.
SQLAlchemy ORM models for data persistence.
"""

# TODO: Import SQLAlchemy components
# from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
# from sqlalchemy.orm import relationship
# from app.database import Base

# TODO: Upload model
# - id (primary key)
# - filename
# - file_path
# - file_type (pdf, image, audio)
# - upload_time
# - status (pending, processing, completed, failed)

# TODO: ProcessedFile model
# - id (primary key)
# - upload_id (foreign key to Upload)
# - output_path
# - processing_time
# - status
# - metadata (JSON field for additional info)

# TODO: ProcessingLog model
# - id (primary key)
# - upload_id (foreign key to Upload)
# - message
# - level (INFO, WARNING, ERROR)
# - timestamp

# TODO: Add relationships between models
# - Upload -> ProcessedFile (one-to-many)
# - Upload -> ProcessingLog (one-to-many)

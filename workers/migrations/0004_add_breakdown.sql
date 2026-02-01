-- Add breakdown column to evaluations table for storing per-rubric-item scores
ALTER TABLE evaluations ADD COLUMN breakdown TEXT DEFAULT NULL;

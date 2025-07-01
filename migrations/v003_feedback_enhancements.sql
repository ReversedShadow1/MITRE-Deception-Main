-- Add to migrations/v002_analysis_tables.sql or create a new migration file

-- Analysis Feedback Table with enhanced fields
CREATE TABLE analysis_feedback (
    id SERIAL PRIMARY KEY,
    analysis_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    technique_id VARCHAR(50) NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id),
    feedback_type VARCHAR(20) NOT NULL, -- 'correct', 'incorrect', 'unsure'
    suggested_alternative VARCHAR(50),
    confidence_level INTEGER, -- Analyst confidence rating (1-5)
    justification TEXT, -- Text explaining the feedback decision
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_feedback_analysis_id ON analysis_feedback(analysis_id);
CREATE INDEX idx_analysis_feedback_user_id ON analysis_feedback(user_id);

-- Table for storing text highlights that justify technique attribution
CREATE TABLE feedback_highlights (
    id SERIAL PRIMARY KEY,
    feedback_id INTEGER NOT NULL REFERENCES analysis_feedback(id) ON DELETE CASCADE,
    segment_text TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_feedback_highlights_feedback_id ON feedback_highlights(feedback_id);

-- Model Training Logs
CREATE TABLE model_training_logs (
    id SERIAL PRIMARY KEY,
    training_data_path VARCHAR(255) NOT NULL,
    example_count INTEGER NOT NULL,
    last_training_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL, -- 'started', 'completed', 'failed'
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_training_logs_status ON model_training_logs(status);
CREATE INDEX idx_model_training_logs_date ON model_training_logs(last_training_date);
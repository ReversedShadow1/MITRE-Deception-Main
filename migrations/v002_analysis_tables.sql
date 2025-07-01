-- migrations/v002_analysis_tables.sql
-- Analysis Jobs (tracking CTI analysis requests)
CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(200), -- Optional name for the analysis job
    status VARCHAR(50) NOT NULL, -- pending, processing, completed, failed
    input_type VARCHAR(50) NOT NULL, -- text, file, url
    input_data TEXT,
    extractors_used TEXT[], -- Array of extractor names used
    threshold FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    processing_time_ms INTEGER
);

CREATE INDEX idx_analysis_jobs_user_id ON analysis_jobs(user_id);
CREATE INDEX idx_analysis_jobs_status ON analysis_jobs(status);
CREATE INDEX idx_analysis_jobs_created_at ON analysis_jobs(created_at);

-- Analysis Results (storing extracted techniques per job)
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    technique_id VARCHAR(50) NOT NULL, -- ATT&CK technique ID (T1234)
    technique_name VARCHAR(200),
    confidence FLOAT NOT NULL,
    method VARCHAR(50) NOT NULL, -- extraction method used
    matched_keywords TEXT[], -- Keywords that matched
    cve_id VARCHAR(50), -- If related to a CVE
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_results_job_id ON analysis_results(job_id);
CREATE INDEX idx_analysis_results_technique_id ON analysis_results(technique_id);
CREATE INDEX idx_analysis_results_confidence ON analysis_results(confidence);

-- Link analysis jobs to MITRE Navigator layers
CREATE TABLE job_layer_links (
    id SERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    layer_id UUID NOT NULL REFERENCES layers(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, layer_id)
);

CREATE INDEX idx_job_layer_links_job_id ON job_layer_links(job_id);
CREATE INDEX idx_job_layer_links_layer_id ON job_layer_links(layer_id);

-- User Dashboard Metrics (for showing statistics on user dashboard)
CREATE TABLE user_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL, -- e.g., 'top_techniques', 'analysis_count'
    metric_value JSONB,
    time_period VARCHAR(50), -- daily, weekly, monthly
    date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_metrics_user_id ON user_metrics(user_id);
CREATE INDEX idx_user_metrics_metric_type ON user_metrics(metric_type);
CREATE INDEX idx_user_metrics_date ON user_metrics(date);

-- API Usage Tracking (for tracking API calls per user)
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX idx_api_usage_created_at ON api_usage(created_at);
CREATE INDEX idx_api_usage_endpoint ON api_usage(endpoint);

-- Analysis Job Bookmarks (for users to save interesting analyses)
CREATE TABLE analysis_bookmarks (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    notes TEXT, -- User notes about why they bookmarked this analysis
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, job_id)
);

CREATE INDEX idx_analysis_bookmarks_user_id ON analysis_bookmarks(user_id);
CREATE INDEX idx_analysis_bookmarks_job_id ON analysis_bookmarks(job_id);

-- Create a view for quick access to user analysis statistics
CREATE VIEW user_analysis_stats AS
SELECT 
    u.id as user_id,
    u.username,
    COUNT(aj.id) as total_analyses,
    COUNT(CASE WHEN aj.status = 'completed' THEN 1 END) as completed_analyses,
    COUNT(CASE WHEN aj.status = 'failed' THEN 1 END) as failed_analyses,
    COUNT(ab.id) as bookmarked_analyses,
    COUNT(DISTINCT ar.technique_id) as unique_techniques_found,
    AVG(ar.confidence) as avg_confidence,
    COUNT(jll.id) as linked_layers
FROM users u
LEFT JOIN analysis_jobs aj ON u.id = aj.user_id
LEFT JOIN analysis_results ar ON aj.id = ar.job_id
LEFT JOIN analysis_bookmarks ab ON u.id = ab.user_id
LEFT JOIN job_layer_links jll ON aj.id = jll.job_id
GROUP BY u.id, u.username;
-- migrations/v004_detailed_metrics.sql

-- Store text segments and their embeddings
CREATE TABLE analysis_text_segments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    segment_text TEXT NOT NULL,
    segment_index INTEGER NOT NULL, -- For ordering
    segment_embedding BYTEA, -- For storing vector embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_text_segments_job_id ON analysis_text_segments(job_id);

-- Store detailed extractor results
CREATE TABLE extractor_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_name VARCHAR(100) NOT NULL, -- e.g., 'rule_based', 'bm25', etc.
    raw_input TEXT, -- What was sent to the extractor
    raw_output JSONB, -- Full output from extractor
    execution_time_ms INTEGER,
    parameters JSONB, -- Parameters used for this extractor
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_extractor_results_job_id ON extractor_results(job_id);
CREATE INDEX idx_extractor_results_extractor ON extractor_results(extractor_name);

-- Store identified entities (NER results)
CREATE TABLE analysis_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type VARCHAR(100) NOT NULL, -- e.g., 'ATTACK-PATTERN', 'MALWARE', etc.
    start_offset INTEGER,
    end_offset INTEGER,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_entities_job_id ON analysis_entities(job_id);
CREATE INDEX idx_analysis_entities_extractor_id ON analysis_entities(extractor_id);

-- Store identified keywords
CREATE TABLE analysis_keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    technique_id VARCHAR(50) NOT NULL, -- Technique the keyword is associated with
    keyword TEXT NOT NULL,
    match_position INTEGER, -- If available
    match_context TEXT, -- Surrounding text for context
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_keywords_job_id ON analysis_keywords(job_id);
CREATE INDEX idx_analysis_keywords_extractor_id ON analysis_keywords(extractor_id);
CREATE INDEX idx_analysis_keywords_technique_id ON analysis_keywords(technique_id);

-- Store BM25 scores
CREATE TABLE bm25_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    technique_id VARCHAR(50) NOT NULL,
    raw_score FLOAT NOT NULL,
    normalized_score FLOAT NOT NULL,
    matched_terms JSONB, -- Terms that matched
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_bm25_scores_job_id ON bm25_scores(job_id);
CREATE INDEX idx_bm25_scores_technique_id ON bm25_scores(technique_id);

-- Store semantic similarity scores
CREATE TABLE semantic_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    technique_id VARCHAR(50) NOT NULL,
    similarity_score FLOAT NOT NULL,
    embedding_dimension INTEGER,
    model_used VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_semantic_scores_job_id ON semantic_scores(job_id);
CREATE INDEX idx_semantic_scores_technique_id ON semantic_scores(technique_id);

-- Store ensemble method details
CREATE TABLE ensemble_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    technique_id VARCHAR(50) NOT NULL,
    ensemble_method VARCHAR(100) NOT NULL, -- e.g., 'advanced_ensemble'
    final_confidence FLOAT NOT NULL,
    component_scores JSONB, -- Scores from each extractor
    weights_used JSONB, -- Weights used in ensemble
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ensemble_details_job_id ON ensemble_details(job_id);
CREATE INDEX idx_ensemble_details_technique_id ON ensemble_details(technique_id);

-- Store Neo4j relationship data
CREATE TABLE technique_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    source_technique_id VARCHAR(50) NOT NULL,
    relationship_type VARCHAR(100) NOT NULL,
    target_entity_type VARCHAR(100) NOT NULL, -- e.g., 'AttackTechnique', 'CVE'
    target_entity_id VARCHAR(100) NOT NULL,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_technique_relationships_job_id ON technique_relationships(job_id);
CREATE INDEX idx_technique_relationships_source_id ON technique_relationships(source_technique_id);

-- Store model metrics and parameters used
CREATE TABLE model_execution_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100),
    device_used VARCHAR(50), -- e.g., 'cpu', 'cuda'
    memory_usage_mb FLOAT,
    batch_size INTEGER,
    quantization_used BOOLEAN,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_execution_metrics_job_id ON model_execution_metrics(job_id);
CREATE INDEX idx_model_execution_metrics_extractor_id ON model_execution_metrics(extractor_id);

-- Store preprocessing details
CREATE TABLE preprocessing_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    original_content_type VARCHAR(50), -- e.g., 'text', 'html', 'pdf'
    tokenization_method VARCHAR(100),
    language_detected VARCHAR(50),
    translated BOOLEAN DEFAULT FALSE,
    normalized BOOLEAN DEFAULT FALSE,
    abbreviations_expanded BOOLEAN DEFAULT FALSE,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preprocessing_details_job_id ON preprocessing_details(job_id);

-- Store NER-specific metrics
CREATE TABLE ner_extraction_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    entity_count INTEGER,
    entity_types JSONB, -- Count of each entity type
    model_name VARCHAR(255),
    aggregation_strategy VARCHAR(100),
    tokenizer_max_length INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ner_extraction_details_job_id ON ner_extraction_details(job_id);

-- Store embedding storage and computation details
CREATE TABLE embedding_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    text_segment_id UUID REFERENCES analysis_text_segments(id) ON DELETE SET NULL,
    technique_id VARCHAR(50),
    embedding_type VARCHAR(100), -- 'query', 'technique', etc.
    embedding_model VARCHAR(255),
    embedding_dimension INTEGER,
    normalization_applied BOOLEAN,
    cache_hit BOOLEAN,
    cosine_similarity FLOAT,
    approximate_search_used BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embedding_details_job_id ON embedding_details(job_id);
CREATE INDEX idx_embedding_details_technique_id ON embedding_details(technique_id);

-- Store KEV (Known Exploited Vulnerabilities) extraction details
CREATE TABLE kev_extraction_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    cve_id VARCHAR(50) NOT NULL,
    cve_mention_position INTEGER,
    cve_mention_context TEXT,
    kev_entry_date DATE, -- When the CVE was added to KEV catalog
    technique_mappings JSONB, -- Techniques mapped to this CVE
    confidence_scores JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kev_extraction_details_job_id ON kev_extraction_details(job_id);
CREATE INDEX idx_kev_extraction_details_cve_id ON kev_extraction_details(cve_id);

-- Store classifier-specific metrics
CREATE TABLE classifier_details (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE,
    model_type VARCHAR(100), -- 'transformer', 'random_forest', etc.
    feature_count INTEGER,
    class_count INTEGER,
    probability_scores JSONB, -- Raw probability scores
    decision_threshold FLOAT,
    embedding_used BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_classifier_details_job_id ON classifier_details(job_id);

-- Store model weights for enhanced tracking
CREATE TABLE model_weights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_id UUID REFERENCES extractor_results(id) ON DELETE CASCADE, 
    model_name VARCHAR(255) NOT NULL,
    weight_path VARCHAR(255),
    weight_hash VARCHAR(64), -- Hash to track weight versions
    weight_size_bytes BIGINT,
    quantization_bits INTEGER, -- 8, 16, 32, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_weights_job_id ON model_weights(job_id);
CREATE INDEX idx_model_weights_model_name ON model_weights(model_name);

-- Store performance benchmarks
CREATE TABLE performance_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    extractor_name VARCHAR(100) NOT NULL,
    operation_type VARCHAR(100) NOT NULL, -- 'tokenization', 'embedding', 'inference', etc.
    input_size INTEGER,
    execution_time_ms INTEGER,
    throughput_tokens_per_second FLOAT,
    memory_peak_mb FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_benchmarks_job_id ON performance_benchmarks(job_id);
CREATE INDEX idx_performance_benchmarks_operation_type ON performance_benchmarks(operation_type);
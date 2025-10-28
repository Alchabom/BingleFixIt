CREATE TABLE agent_comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rating INT NOT NULL,
    review_content TEXT NOT NULL,
    episode_id INT NOT NULL,
    step_number INT NOT NULL,
    reward FLOAT NOT NULL,
    action_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_episode_step (episode_id, step_number),
    INDEX idx_rating (rating),
    INDEX idx_created (created_at)
);
-- Run this once in IBM Db2 / Db2 on Cloud "Run SQL" (as your app user, e.g. 60676f88).
-- Then restart Sensus. Alternatively set IBM_DB_AUTO_SCHEMA=1 in .env for dev only.

CREATE TABLE sessions (
  id VARCHAR(36) NOT NULL PRIMARY KEY,
  user_id VARCHAR(36),
  name VARCHAR(255),
  created_at TIMESTAMP,
  summary CLOB
);

CREATE TABLE messages (
  id VARCHAR(36) NOT NULL PRIMARY KEY,
  session_id VARCHAR(36),
  role VARCHAR(16),
  content CLOB,
  agent VARCHAR(32),
  created_at TIMESTAMP
);

CREATE TABLE preferences (
  user_id VARCHAR(36) NOT NULL PRIMARY KEY,
  prefs_json CLOB
);

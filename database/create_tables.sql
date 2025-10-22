-- participants
CREATE TABLE `participants` (
    `uuid` CHAR(36) NOT NULL,
    `name` varchar(255) DEFAULT NULL,
    `type` varchar(255) DEFAULT NULL,
    `sex` varchar(255) DEFAULT NULL,
    `date_of_birth` DATETIME DEFAULT NULL,
  PRIMARY KEY (`uuid`)
);

-- sessions
CREATE TABLE `sessions` (
    `uuid` CHAR(36) NOT NULL,
    `name` varchar(255) DEFAULT NULL,
    `type` varchar(255) DEFAULT NULL,
    `participant_id` CHAR(136) NOT NULL,
  PRIMARY KEY (`uuid`),
  FOREIGN KEY (`participant_id`) REFERENCES participants(`uuid`) ON DELETE CASCADE
);

-- qtm_data
CREATE TABLE qtm_data (
    `uuid` CHAR(36) NOT NULL,
    `session_id` CHAR(36) NOT NULL,
    `file` VARCHAR(255) DEFAULT NULL,
    `trial` INT(11) DEFAULT NULL,
    `repetition` INT(11) DEFAULT NULL,
    `type` VARCHAR(255) DEFAULT NULL,
    `start_time` DATETIME DEFAULT NULL,
    `valid` TINYINT(1) DEFAULT 1,
    PRIMARY KEY (`uuid`),
    FOREIGN KEY (`session_id`) REFERENCES sessions(`uuid`) ON DELETE CASCADE
);

-- View combining qtm_data, sessions, and participants
CREATE VIEW qtm_full_view AS
SELECT
    qtm_data.uuid AS qtm_id,
    qtm_data.session_id,
    sessions.participant_id,
    qtm_data.file,
    qtm_data.trial,
    qtm_data.repetition,
    qtm_data.type AS trial_type,
    qtm_data.start_time,
    qtm_data.valid,
    sessions.name AS session_name,
    sessions.type AS session_type,
    participants.name AS participant_name,
    participants.type AS participant_type,
    participants.sex,
    participants.date_of_birth
FROM qtm_data
JOIN sessions ON qtm_data.session_id = sessions.uuid
JOIN participants ON sessions.participant_id = participants.uuid;
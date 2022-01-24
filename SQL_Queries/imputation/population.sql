-- Extract "static" patient information: demographics, date of birth and
-- death, date of ICU admission and so on
select distinct
    i.subject_id,           -- patient identifier
    i.hadm_id,              -- hospital admission identifier (patient may have >1 admission)
    i.icustay_id,           -- ICU stay identifier (patient may have >1 of these; if so, they will be in increasing order)
    i.gender,               -- M or F
    i.admission_age as age, -- age at admission (as float, not whole years)
    i.hospital_expire_flag, -- 0 or 1: This indicates whether the patient died within the given hospitalization. 1 indicates death in the hospital, and 0 indicates survival to hospital discharge.
    i.hospstay_seq,         -- 1, 2, 3, ... - which visit to the ICU is this for this patient?
    i.los_icu,              -- length of stay in ICU in days (floating point)
    i.admittime,            -- datetime of admission to hospital
    i.dischtime,            -- datetime of discharge from hospital; if died in hospital, this should be the same as deathtime
    i.intime,               -- datetime of ICU admission
    i.outtime,              -- datetime of ICU discharge (or death)
    a.admission_type,       -- ELECTIVE, URGENT, NEWBORN or EMERGENCY
    a.deathtime,            -- datetime of in-hospital death, otherwise null
    i.dod,                  -- date of death if known, otherwise null
    p.expire_flag,          -- whether the patient death date is known
    CASE when a.deathtime between i.intime and i.outtime THEN 1 ELSE 0 END AS mort_icu                      -- died during ICU stay
FROM icustay_detail i  -- from mimic-iii/concepts/demographics/icustay_detail.sql
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN patients p ON i.subject_id = p.subject_id
WHERE i.hadm_id is not null and i.icustay_id is not null
    and i.hospstay_seq = 1
    and i.icustay_seq = 1
    and i.admission_age >= {min_age}
    and i.los_icu >= {min_day}
    and i.outtime >= (i.intime + interval '{min_dur} hours')
    and a.admission_type in ({admission_types})
ORDER BY subject_id
{limit}

-- Extract clinical information from the ICU stay
select
    c.subject_id,  -- patient identifier
    c.hadm_id,     -- hospital admission identifier
    c.icustay_id,  -- ICU stay identifier
    c.charttime,   -- time of event as recorded on chart
    c.itemid,      -- identifier for measurement type
    c.valuenum,    -- measurement value (only if numeric value)
    valueuom       -- unit of measurement (if applicable)
FROM icustay_detail i
    INNER JOIN chartevents c ON i.icustay_id = c.icustay_id
    WHERE c.icustay_id in ({icuids})
      and c.itemid in ({chitem})
      and c.charttime between i.intime and i.outtime
      and c.error is distinct from 1
      and c.valuenum is not null

--DROP SCHEMA public CASCADE;
--
--CREATE SCHEMA public;
--GRANT ALL ON SCHEMA public TO postgres;
--GRANT ALL ON SCHEMA public TO public;

--SELECT trigger_name,
--       event_object_table AS table_name,
--       event_manipulation AS event,
--       action_timing AS timing,
--       action_statement AS definition
--FROM information_schema.triggers;
--
--ALTER TABLE training_einstellung ENABLE TRIGGER training_einstellung_tr;

--#TRAINING_EINSTELLUNG###############################--
--####################################################--
create table training_einstellung(
	id 				integer NOT NULL,
	num_epochs 		integer,
	batch_size 		integer,
	seed 			integer,
	noising 		double precision,
	clipping 		double precision,
	model 			varchar(200),
	dataset 		varchar(200),
	make_private 	integer,
	data_processing integer,
	dataset_mode 	integer,
	init_mode 		integer,
	dp_mode			integer,
	num_trainings 	integer,
	length_dataset 	integer,
	learning_rate 	double precision,
	beschreibung 	varchar(200),
	constraint 		training_einstellung_pk primary key (id)
);

create sequence training_einstellung_seq
start with 1
increment by 1;

create or replace function set_training_einstellung_id()
returns trigger as $$
begin
    new.id := nextval('training_einstellung_seq');
    return new;
end;
$$ language plpgsql;

create or replace trigger training_einstellung_tr
before insert on training_einstellung
for each row
execute function set_training_einstellung_id();


--#TRAINING###########################################--
--####################################################--
create table training(
	run			 		integer,
	training 			integer,
	loss 				double precision,
	accuracy 			double precision,
	epsilon 			double precision,
	epsilon_batching	double precision,
	model_name 			varchar(200),
	zeitpunkt 			timestamp,
	constraint			training_pk primary key(run,training),
	constraint 			training_fk foreign key(run) references training_einstellung(id) on delete cascade
);

--#distance########################################--
--####################################################--
create table distance(
	run				integer,
	model1			integer,
	model2			integer,
	distance		double precision,
	constraint		distance_pk primary key (run,model1, model2),
	constraint		distance_fk foreign key(run) references training_einstellung(id) on delete cascade
);

--#STATISTIKEN MATERIALIZED VIEW######################--
--####################################################--
create materialized view statistiken as
with prepared_stats as (
    select	 run
        	 ,avg(distance) as avg_value
        	 ,stddev(distance) as stddev_value
    from 	 distance
    group by run
)
select	 	run
    		,min(distance) as minimum
    		,max(distance) as maximum
    		,avg(distance) as durschschnitt
    		,percentile_cont(0.5) within group (order by distance) as median
    		,max(distance) - min(distance) as spannweite
    		,variance(distance) as varianz
    		,stddev(distance) as standardabweichung
    		,percentile_cont(0.75) within group (order by distance) - percentile_cont(0.25) within group (order by distance) as interquartilsabstand
    		,(count(distance) * sum(power(distance - (select avg_value from prepared_stats where prepared_stats.run = distance.run), 3))) / ((count(distance) - 1) * (count(distance) - 2) * power((select stddev_value from prepared_stats where prepared_stats.run = distance.run), 3)) as schiefe
    		,(count(distance) * (count(distance) + 1) * sum(power(distance - (select avg_value from prepared_stats where prepared_stats.run = distance.run), 4))) / ((count(distance) - 1) * (count(distance) - 2) * (count(distance) - 3) * power((select stddev_value from prepared_stats where prepared_stats.run = distance.run), 4)) - (3 * power(count(distance) - 1, 2)) / ((count(distance) - 2) * (count(distance) - 3)) as wölbung
from 		distance
group by 	run;
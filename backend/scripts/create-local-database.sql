select 'create role edu_rag login password ''edu_rag_local'''
where not exists (select 1 from pg_roles where rolname = 'edu_rag') \gexec

alter role edu_rag with login password 'edu_rag_local';

select 'create database edu_rag owner edu_rag'
where not exists (select 1 from pg_database where datname = 'edu_rag') \gexec

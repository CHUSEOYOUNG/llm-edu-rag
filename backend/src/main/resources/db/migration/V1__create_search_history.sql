create table search_history (
    id uuid primary key,
    client_id uuid not null,
    question varchar(500) not null,
    search_query varchar(1000) not null,
    school_level varchar(20) not null,
    result_count integer not null,
    created_at timestamp with time zone not null
);

create index idx_search_history_client_created
    on search_history (client_id, created_at desc);

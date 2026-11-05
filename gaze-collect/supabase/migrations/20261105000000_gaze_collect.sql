-- gaze-collect: calibration sessions from invited participants.
-- Additive only: new gaze_* tables and one private bucket. Nothing existing is touched.
-- No policies on purpose: row-level security is on, so the browser's anon key can read or write nothing here.
-- All access goes through the gaze-upload edge function and the owner's sync script (service role).

create table public.gaze_participants (
  id uuid primary key default gen_random_uuid(),
  alias text not null unique,                 -- P01, P02, ...; the code-to-name list stays on the owner's PC
  code_hash text not null unique,             -- SHA-256 of the invite code
  active boolean not null default true,
  consent_version text,
  consent_at timestamptz,
  created_at timestamptz not null default now()
);

create table public.gaze_sessions (
  id uuid primary key default gen_random_uuid(),
  participant_id uuid not null references public.gaze_participants (id) on delete cascade,
  client_id text not null,                    -- makes retries of the same session idempotent
  status text not null default 'uploading' check (status in ('uploading', 'uploaded', 'synced')),
  size_bytes bigint not null check (size_bytes > 0),
  part_count integer not null check (part_count > 0),
  sha256 text not null,
  lighting text not null,
  created_at timestamptz not null default now(),
  uploaded_at timestamptz,
  synced_at timestamptz,
  unique (participant_id, client_id)
);

create index gaze_sessions_status on public.gaze_sessions (status);

alter table public.gaze_participants enable row level security;
alter table public.gaze_sessions enable row level security;

revoke all on public.gaze_participants from anon, authenticated;
revoke all on public.gaze_sessions from anon, authenticated;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('gaze-sessions', 'gaze-sessions', false, 26214400, array['application/octet-stream']);

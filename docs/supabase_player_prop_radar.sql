create table if not exists public.player_prop_radar (
  signal_id text primary key,
  match_number integer,
  match_date text,
  phase text,
  home_team text,
  away_team text,
  team text,
  opponent text,
  player_key text,
  player_name text,
  position_broad text,
  market text,
  market_label text,
  line double precision,
  selection text,
  model_probability double precision,
  fair_odds double precision,
  min_odds_review double precision,
  expected_minutes double precision,
  lineup_status text,
  lineup_rule text,
  mobile_action text,
  action_if_starter text,
  action_if_bench text,
  tracking_action text,
  data_quality_tier text,
  paper_tracking_allowed boolean,
  updated_at timestamptz
);

alter table public.player_prop_radar enable row level security;

drop policy if exists "player_prop_radar_read" on public.player_prop_radar;
create policy "player_prop_radar_read"
on public.player_prop_radar
for select
to anon, authenticated
using (true);


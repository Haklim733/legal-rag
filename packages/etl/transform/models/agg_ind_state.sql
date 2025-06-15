MODEL (
  name irs_analytics.agg_ind_state,
  kind VIEW,
  column_descriptions (
      state= "State postal abbreviation",
      num_returns= "Number of individual income tax returns filed",
      total_agi= "Total adjusted gross income (AGI)",
      row_count= "Number of rows in the data",
      report_total_agi= "Total AGI from the IRS report",
      diff_total_agi= "Difference between AGI from the data and the report",
      diff_total_agi_pct= "Percentage difference between AGI from the data and the report"
  ),
  audits (
        not_null(columns=[state, num_returns, total_agi, row_count]),
        unique_combination_of_columns(columns=[state])
    ),
grains [
    "state",
],
);

WITH agg AS (
    SELECT
    state,
    SUM(n1),
    SUM(a00100),
    COUNT(*) AS row_count
FROM irs.ind_county 
WHERE county != '000'
GROUP BY state)
SELECT 
    a.state::CHAR(2),
    a.num_returns::INTEGER,
    a.total_agi::BIGINT,
    a.row_count::INTEGER,
    b.total_agi as report_total_agi, 
    a.total_agi - b.total_agi as diff_total_agi, 
    ROUND((a.total_agi - b.total_agi) / b.total_agi, 4) as diff_total_agi_pct 
FROM agg a
LEFT JOIN irs.reports_state b
ON a.state = b.state
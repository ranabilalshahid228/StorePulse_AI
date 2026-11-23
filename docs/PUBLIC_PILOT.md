# Prepare a public StorePulse AI pilot

## Onboard a store

1. Have the store owner create a workspace at the deployed sign-in page.
2. In Data & EDA, upload a CSV as Sales history. Use one row per store and calendar day. A minimal file has
   columns `Date,Store,Sales`. Use ISO dates, a stable store code, and nonnegative daily net sales.
   Include `Open` (0 or 1) for closures, `Customers` for footfall estimates, and `Promo` (0 or 1)
   if scheduled promotions are available. Use store-level totals with no customer identifiers.
3. Map the date, store, and sales columns. If present, map Customers, Open, and Promo to their named roles.
   Set Open and Promo as known-future inputs only if their future values are available.
4. Under Stores, configure currency, cost of goods, fixed overhead, wage, and staffing assumptions.
   For a sales target without customer counts, enter average transaction value. Confirm your future
   opening and promotion assumptions before projecting a plan.
5. In Model Lab, train LightGBM and the weekday baseline on the same store scope and validation window.
   Start with a 28-day forecast and validation window and at least six months of daily history.
   Review validation WAPE before using a forecast for an operating decision.
6. Run a forecast, create a Store Autopilot plan, and check the plan against the store's known hours,
   promotions, costs, and staffing requirements.

Generate a complete synthetic workspace without retailer data by running
`python -m scripts.smoke_local --synthetic --stores 1 2 3` from `backend/` with the stack running.

## Reconcile a retailer's daily export

Start with a read-only CSV export from the pilot store's POS or ERP. Group transactions by store
and local trading date. Agree with the store owner on currency, tax treatment, discounts, and how
returns and late corrections appear. Keep refunds and cancellations in the reconciliation so daily
totals agree with the retailer's report. Investigate negative net-sales days before importing:
the forecasting workflow is designed for nonnegative daily targets.

Use a stable store code. Keep customer names, contact details, and payment information out of the
export. POS receipt counts are transaction counts. Supply measured visitor counts separately if
they are needed for workforce planning.

After mapping, check the **Mapped sales preview** and **Data health** in StorePulse. Reconcile one
normal day, one promotion day, one closed day, and one day with returns against the source report.
For a first pilot, refresh via the retailer's existing export process. Select a live read-only
connector once the retailer confirms its POS or ERP product and provides a test environment.

## Hosting and release checks

- Use TLS on a public domain. Route the frontend and API through a managed ingress or reverse proxy.
  Keep PostgreSQL, Redis, and artifact storage on private networks. The example Compose file binds
  its published service ports to loopback for local development.
- Set `ENVIRONMENT=production`, `DEBUG=false`, a unique random `JWT_SECRET_KEY`, and strong database
  credentials. Set `CORS_ORIGINS` to the exact HTTPS frontend origin and build the frontend with
  `NEXT_PUBLIC_API_URL` pointing at the deployed HTTPS API.
- Use private, persistent storage for datasets and model artifacts. Configure database and artifact
  backups and test a restore before onboarding stores.
- Configure and exercise rate limits for sign-up, sign-in, uploads, and job submissions. Set
  `MAX_UPLOAD_MB`, worker concurrency, and per-account job quotas appropriate to measured capacity.
  Confirm that an expensive training run cannot exhaust resources needed by other stores.
- Run the synthetic smoke test against staging, verify account isolation, and rehearse upload,
  mapping, forecast, Autopilot, and export with a pilot user. Simulate a queue outage and verify
  errors and retries. Monitor failed and stuck jobs, storage growth, and worker health.
- Publish your privacy and retention terms, and verify a process to remove a store's data and artifacts.
  For a pilot, limit access at the ingress until these release checks pass.

Existing saved Autopilot reports are snapshots. Regenerate any report created with the old missing-band
behavior before sharing it.

## Run quick forecasts

Open **Forecasts**, select a completed model, and choose **Next day**, **Next week**, or
**Next 30 days**. The window starts after the dataset's last date. Train with a horizon of at
least 30 days to enable all three buttons. Check the date range and validation WAPE before
using the chart or downloading the CSV.

For product rankings, use one row per trading day, store, and SKU, for example
`Date,Store,SKU,Units`. Map SKU as the item identifier and Units as the target before training.
A revenue target ranks products by predicted revenue. A store-total dataset supports sales
forecasting without SKU rankings.

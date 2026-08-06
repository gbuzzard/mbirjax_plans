# Bouman Group — GPU Compute Resources at Purdue RCAC

*Compiled 2026-08-06 from the RCAC account portal (rcac.purdue.edu/account), order history, and live `slist` on each cluster.*

## Summary

| Cluster  | Resource              | Queue      | Amount            | Hours left  | Ends       |
|----------|-----------------------|------------|-------------------|-------------|------------|
| Gautschi | H100 GPU-hours        | `bouman`   | 32,850 SU         | **2,716.4** | ~2030      |
| Gilbreth | A100 node (dedicated) | `bouman-g` | 2 GPUs, 128 cores | no cap      | ~late 2026 |
| Gilbreth | A100 GPU access       | `bouman-n` | 2 GPUs, 24 cores  | no cap      | Dec 2029   |
| Gilbreth | old annual sub        | `bouman-e` | expired           | —           | Sep 2021   |

*Hours-left figure is as of 2026-08-06; the ~late-2026 node retirement is inferred (see details below).*

Live `slist` on Gilbreth currently reports the group's allocation as **4× A100-40GB total, all free** under account `bouman`.

## Gautschi details (H100, metered by GPU-hours)

- Purchases: two "Quarter-GPU hours for 5 years" orders ($7,813 each, Mar 2025 and Oct 2025) plus an IPAI 1:1 match (Nov 2025, $0). A third quarter-GPU order (#17563, $7,813) is pending business office approval as of Jul 30 2026 and will add hours when it completes.
- A quarter GPU = 2,190 GPU-hours/year.
- Charging: `normal` QOS burns 1 GPU-hour per GPU per hour; `preemptible` QOS burns 0.25× (job can be preempted).
- Submit: `sbatch -A bouman -p ai --gres=gpu:1 ...` (add `-q preemptible` for the 0.25× rate).
- Check balance anytime: `slist` on a Gautschi login node.

## Gilbreth details (A100-40GB, metered by GPU count)

- No hour metering: the subscription caps how many GPUs the group can use simultaneously (4 total), not total usage.
- Purchases: dedicated A100 node, Nov 2021, $17,530 (queue `bouman-g`); 2× "A100-40GB GPU Access (5 years)", Dec 2024, $14,000 (queue `bouman-n`).
- Submit: `sbatch -A <account> -p a100-40gb --gres=gpu:1 ...` — run `slist` on Gilbreth to see the exact account name(s) available to you.
- The dedicated node's retirement date is not shown in the portal; ~late 2026 is inferred from the standard 5-year node lifetime. Confirm with rcac-help@purdue.edu.

## What students need to get access

1. **Purdue career account** with BoilerKey/Duo two-factor.
2. **Be added to the group's queue(s).** Two paths:
   - Student logs into rcac.purdue.edu/account → **Request Access**, requests the relevant queue (`bouman` on Gautschi, `bouman-g`/`bouman-n` on Gilbreth); the PI approves.
   - Or the PI adds them directly: rcac.purdue.edu/account → Groups → Charles Bouman Group → **Members** tab. (RCAC restored the group-manager role Aug 2026 after a portal-migration glitch had hidden these controls.)
3. **Log in** (changes can take ~1 day to propagate, plus a log-out/log-in after group changes):
   - `ssh <username>@gautschi.rcac.purdue.edu` or `ssh <username>@gilbreth.rcac.purdue.edu`
   - Or the web portals (Open OnDemand / Gateway) linked from each cluster's page on rcac.purdue.edu.
4. **Verify access:** run `slist` on the cluster — the group account should appear. Then submit with `-A <account>` as shown above.

## Other compute (non-GPU, for reference)

- Negishi: two 64-core CPU shares ($4,200 each, Sep/Oct 2023), queue `bouman-n`.
- Research Data Depot storage: multiple 1 TB subscriptions, renewed annually.

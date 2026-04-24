# File Curation Notes

Selection policy: keep only final merged and complete repaired versions.

## Kept priority
1. all / all_dedup / all_dedup_repaired over group-level split files.
2. repaired outputs over non-repaired versions.
3. files that form an end-to-end runnable chain.

## Removed categories
- split intermediate files (group*, mor, siam, mp)
- backups (*.bak)
- duplicate old variants
- temporary testing artifacts

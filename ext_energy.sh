#!/usr/bin/env bash
set -euo pipefail

outfile="${1:-all_energy.txt}"
infile="${2:-OUTCAR}"
if [[ ! -f "$infile" ]]; then
    echo "OUTCAR not found: $infile" >&2
    exit 1
fi
tmp=$(mktemp "${outfile}.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT

if ! awk '
/energy  without entropy/ {
    if ($4 !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([Ee][-+]?[0-9]+)?$/) {
        bad = 1; exit 1
    }
    print $4
    count++
}
END { if (bad || !count) exit 1 }
' "$infile" > "$tmp"; then
    echo "Energy extraction failed: missing or malformed energy records in $infile." >&2
    exit 1
fi

mv -- "$tmp" "$outfile"
echo "Extracted all energy-without-entropy values (eV) to $outfile."

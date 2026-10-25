#!/usr/bin/env bash
set -euo pipefail

outfile="${1:-forces_only.dat}"
infile="${2:-OUTCAR}"
natoms="${NATOMS:-80}"
if [[ ! -f "$infile" || ! "$natoms" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: NATOMS=80 bash ext_force.sh [forces_only.dat] [OUTCAR]" >&2
    exit 1
fi
tmp=$(mktemp "${outfile}.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT

if ! awk -v natoms="$natoms" '
/POSITION *TOTAL-FORCE/ {
    if ((getline) <= 0) { bad = 1; exit 1 }
    for (i = 0; i < natoms; i++) {
        if ((getline) <= 0 || NF != 6) { bad = 1; exit 1 }
        for (j = 4; j <= 6; j++) {
            if ($j !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([Ee][-+]?[0-9]+)?$/) {
                bad = 1; exit 1
            }
        }
        print $4, $5, $6
    }
    print ""
    blocks++
}
END { if (bad || !blocks) exit 1 }
' "$infile" > "$tmp"; then
    echo "Force extraction failed: missing, incomplete, or malformed force blocks in $infile." >&2
    exit 1
fi

mv -- "$tmp" "$outfile"
echo "Extracted all force frames to $outfile ($natoms atoms per frame)."

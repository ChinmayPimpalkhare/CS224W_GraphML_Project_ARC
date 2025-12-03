#!/bin/bash
#
# Recompute user profiles with new half-life (20 days instead of 150)
#
# This will regenerate user profiles for both 10% and 25% subsamples
# with faster temporal decay (20 days half-life)
#

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║    Recomputing User Profiles with Updated Half-Life (20 days)        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Old half-life: 150 days"
echo "New half-life: 20 days"
echo ""
echo "Impact: Recent interactions will have MUCH more weight"
echo "  - Interactions from 20 days ago: weight = 0.5"
echo "  - Interactions from 40 days ago: weight = 0.25"
echo "  - Interactions from 60 days ago: weight = 0.125"
echo ""
echo "This will:"
echo "  1. Recompute user profiles for 10% subsample"
echo "  2. Recompute user profiles for 25% subsample"
echo "  3. Update any existing profiles with new decay"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/2: Recomputing 10% subsample user profiles..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -d "data/processed/ml1m_10pct" ]; then
    echo "Found 10% subsample, recomputing profiles..."
    
    # Backup old profiles
    if [ -f "data/processed/ml1m_10pct/user_profiles.pt" ]; then
        echo "Backing up old profiles..."
        mv data/processed/ml1m_10pct/user_profiles.pt \
           data/processed/ml1m_10pct/user_profiles_halflife150.pt.backup
    fi
    
    # Recompute with new half-life
    python scripts/compute_user_profiles.py \
        --data_dir data/processed/ml1m_10pct \
        --half_life_days 20 \
        --seed 42
    
    echo ""
    echo "✅ 10% subsample profiles updated"
else
    echo "⚠️  10% subsample not found, skipping..."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/2: Recomputing 25% subsample user profiles..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -d "data/processed/ml1m_25pct" ]; then
    echo "Found 25% subsample, recomputing profiles..."
    
    # Backup old profiles
    if [ -f "data/processed/ml1m_25pct/user_profiles.pt" ]; then
        echo "Backing up old profiles..."
        mv data/processed/ml1m_25pct/user_profiles.pt \
           data/processed/ml1m_25pct/user_profiles_halflife150.pt.backup
    fi
    
    # Recompute with new half-life
    python scripts/compute_user_profiles.py \
        --data_dir data/processed/ml1m_25pct \
        --half_life_days 20 \
        --seed 42
    
    echo ""
    echo "✅ 25% subsample profiles updated"
else
    echo "⚠️  25% subsample not found, skipping..."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                          ✅ ALL DONE!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "User profiles have been recomputed with half-life = 20 days"
echo ""
echo "Backups saved as:"
echo "  • user_profiles_halflife150.pt.backup"
echo ""
echo "Impact on training:"
echo "  ✅ Recent interactions weighted more heavily"
echo "  ✅ Better captures current user preferences"
echo "  ✅ Should improve recommendation quality"
echo ""
echo "Note: Currently running training (25% subsample) is using OLD profiles."
echo "To use new profiles, you need to restart training:"
echo ""
echo "  1. Stop current training (Ctrl+C in screen session)"
echo "  2. Rerun: ./train_25pct_proper.sh"
echo ""
echo "Or wait for current training to complete, then train again with new profiles."
echo ""

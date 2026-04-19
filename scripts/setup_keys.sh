#!/bin/bash
# setup_keys.sh
# Safely store and load API keys for the Northern Thai LLM project.
# Keys are saved to ~/.northern_thai_keys (never committed to GitHub)
#
# Usage:
#   bash scripts/setup_keys.sh set      <- enter your keys interactively
#   bash scripts/setup_keys.sh load     <- load keys into current session
#   bash scripts/setup_keys.sh show     <- show which keys are set (masked)
#   bash scripts/setup_keys.sh clear    <- delete all saved keys

KEYS_FILE="$HOME/.northern_thai_keys"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helper: mask key for display ──────────────────────────────────────────────
mask_key() {
    local key="$1"
    if [ -z "$key" ] || [ "$key" = "not set" ]; then
        echo "not set"
    else
        # Show first 6 and last 4 characters only
        echo "${key:0:6}...${key: -4}"
    fi
}

# ── Set keys interactively ────────────────────────────────────────────────────
set_keys() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} Northern Thai LLM — API Key Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${YELLOW}Keys are saved to: $KEYS_FILE${NC}"
    echo -e "${YELLOW}Press Enter to skip a key and keep existing value.${NC}"
    echo ""

    # Load existing keys if file exists
    if [ -f "$KEYS_FILE" ]; then
        source "$KEYS_FILE"
    fi

    # Prompt for each key
    echo -e "${GREEN}OpenAI API Key${NC} (platform.openai.com)"
    echo -e "Current: $(mask_key "$OPENAI_API_KEY")"
    read -s -p "New key (or Enter to skip): " input
    echo ""
    if [ -n "$input" ]; then OPENAI_API_KEY="$input"; fi

    echo ""
    echo -e "${GREEN}Anthropic API Key${NC} (console.anthropic.com)"
    echo -e "Current: $(mask_key "$ANTHROPIC_API_KEY")"
    read -s -p "New key (or Enter to skip): " input
    echo ""
    if [ -n "$input" ]; then ANTHROPIC_API_KEY="$input"; fi

    echo ""
    echo -e "${GREEN}Google Gemini API Key${NC} (aistudio.google.com)"
    echo -e "Current: $(mask_key "$GEMINI_API_KEY")"
    read -s -p "New key (or Enter to skip): " input
    echo ""
    if [ -n "$input" ]; then GEMINI_API_KEY="$input"; fi

    echo ""
    echo -e "${GREEN}DeepSeek API Key${NC} (platform.deepseek.com)"
    echo -e "Current: $(mask_key "$DEEPSEEK_API_KEY")"
    read -s -p "New key (or Enter to skip): " input
    echo ""
    if [ -n "$input" ]; then DEEPSEEK_API_KEY="$input"; fi

    echo ""
    echo -e "${GREEN}ThaiLLM API Key${NC} (playground.thaillm.or.th)"
    echo -e "Current: $(mask_key "$THAILLM_API_KEY")"
    read -s -p "New key (or Enter to skip): " input
    echo ""
    if [ -n "$input" ]; then THAILLM_API_KEY="$input"; fi

    # Write to file
    cat > "$KEYS_FILE" << EOF
# Northern Thai LLM Project — API Keys
# Generated: $(date)
# DO NOT commit this file to GitHub

export OPENAI_API_KEY="$OPENAI_API_KEY"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export GEMINI_API_KEY="$GEMINI_API_KEY"
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export THAILLM_API_KEY="$THAILLM_API_KEY"
EOF

    # Secure the file — only you can read it
    chmod 600 "$KEYS_FILE"

    echo ""
    echo -e "${GREEN}Keys saved to $KEYS_FILE${NC}"
    echo -e "${YELLOW}To load them into your session run:${NC}"
    echo -e "  source $KEYS_FILE"
    echo ""
    echo -e "${YELLOW}Or add this to your ~/.zprofile to load automatically:${NC}"
    echo -e "  source $KEYS_FILE"
}

# ── Load keys into current session ────────────────────────────────────────────
load_keys() {
    if [ ! -f "$KEYS_FILE" ]; then
        echo -e "${RED}No keys file found. Run: bash scripts/setup_keys.sh set${NC}"
        exit 1
    fi
    source "$KEYS_FILE"
    echo -e "${GREEN}Keys loaded into current session:${NC}"
    show_keys
}

# ── Show which keys are set ───────────────────────────────────────────────────
show_keys() {
    if [ -f "$KEYS_FILE" ]; then
        source "$KEYS_FILE"
    fi

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} API Key Status${NC}"
    echo -e "${BLUE}========================================${NC}"

    keys=(
        "OPENAI_API_KEY:GPT-4o (platform.openai.com)"
        "ANTHROPIC_API_KEY:Claude (console.anthropic.com)"
        "GEMINI_API_KEY:Gemini (aistudio.google.com)"
        "DEEPSEEK_API_KEY:DeepSeek (platform.deepseek.com)"
        "THAILLM_API_KEY:ThaiLLM (playground.thaillm.or.th)"
    )

    for entry in "${keys[@]}"; do
        var="${entry%%:*}"
        label="${entry##*:}"
        val="${!var}"
        if [ -n "$val" ]; then
            echo -e "  ${GREEN}✓${NC} $label"
            echo -e "    ${val:0:6}...${val: -4}"
        else
            echo -e "  ${RED}✗${NC} $label — not set"
        fi
    done
    echo ""
}

# ── Clear all keys ────────────────────────────────────────────────────────────
clear_keys() {
    if [ -f "$KEYS_FILE" ]; then
        rm "$KEYS_FILE"
        echo -e "${GREEN}Keys file deleted.${NC}"
    else
        echo -e "${YELLOW}No keys file found.${NC}"
    fi
}

# ── Run evaluation with all available keys ────────────────────────────────────
run_eval() {
    if [ ! -f "$KEYS_FILE" ]; then
        echo -e "${RED}No keys found. Run: bash scripts/setup_keys.sh set${NC}"
        exit 1
    fi
    source "$KEYS_FILE"

    echo -e "${BLUE}Running translation evaluation with available keys...${NC}"
    echo ""

    python scripts/evaluate_translation_api.py \
        --openai_key    "$OPENAI_API_KEY" \
        --anthropic_key "$ANTHROPIC_API_KEY" \
        --gemini_key    "$GEMINI_API_KEY" \
        --deepseek_key  "$DEEPSEEK_API_KEY" \
        --thaillm_key   "$THAILLM_API_KEY"
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "$1" in
    set)    set_keys ;;
    load)   load_keys ;;
    show)   show_keys ;;
    clear)  clear_keys ;;
    run)    run_eval ;;
    *)
        echo -e "${BLUE}Northern Thai LLM — Key Manager${NC}"
        echo ""
        echo "Usage:"
        echo "  bash scripts/setup_keys.sh set    <- enter your keys interactively"
        echo "  bash scripts/setup_keys.sh show   <- show which keys are set"
        echo "  bash scripts/setup_keys.sh load   <- load keys into session"
        echo "  bash scripts/setup_keys.sh run    <- run evaluation with saved keys"
        echo "  bash scripts/setup_keys.sh clear  <- delete all saved keys"
        ;;
esac

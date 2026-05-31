#!/usr/bin/env bash
# =============================================================================
# setup_remote.sh — One-shot infrastructure setup for VecAdvisor++ experiments
#
# Run this script on the remote server from the project root:
#   cd /tmp2/b11902156/VecAdvisor-pp
#   bash scripts/setup_remote.sh
#
# What it does:
#   0. Downloads PostgreSQL server binaries if missing (from Arch archive)
#   1. Initialises a personal PostgreSQL 18 cluster at $PGDATA (port 15432)
#   2. Compiles and installs pgvector from source
#   3. Installs Rust toolchain + maturin, builds vecadvisor_rs extension
#   4. Downloads SIFT1M and GIST1M datasets
#
# No credentials are stored in this script. The PostgreSQL cluster uses
# OS-level trust authentication (initdb default).
# =============================================================================
set -euo pipefail

REMOTE_BASE=/tmp2/b11902156
PGDATA=$REMOTE_BASE/pgdata
PGPORT=15432
PGDB=vecadvisor
PROJECT_DIR=$REMOTE_BASE/VecAdvisor-pp
DATA_DIR=$REMOTE_BASE/data
PG_SERVER_DIR=$REMOTE_BASE/pg_server

# ---------------------------------------------------------------------------
# Step 0: Ensure PostgreSQL server binaries are available
# ---------------------------------------------------------------------------
# On Arch Linux, server tools (initdb, pg_ctl, postgres) live in the
# 'postgresql' package, which may not be installed system-wide. We download
# the Arch package from the archive and extract binaries locally.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 0: PostgreSQL server binaries ---"

_ensure_pg_server_binaries() {
    if command -v initdb &>/dev/null; then
        echo "initdb found in PATH: $(which initdb)"
        return
    fi

    if [ -x "$PG_SERVER_DIR/usr/bin/initdb" ]; then
        echo "Using previously extracted PostgreSQL server binaries."
        return
    fi

    echo "PostgreSQL server binaries not found. Downloading Arch package..."
    mkdir -p "$PG_SERVER_DIR"

    # Match the installed postgresql-libs version for ICU compatibility.
    # The system has postgresql-libs 18.0-1 which links against ICU 76.
    # We must download a package built against the same ICU version.
    local pkg_url="https://archive.archlinux.org/packages/p/postgresql/postgresql-18.0-1-x86_64.pkg.tar.zst"
    curl -L -o "$PG_SERVER_DIR/postgresql.pkg.tar.zst" "$pkg_url"

    local fsize
    fsize=$(stat -c%s "$PG_SERVER_DIR/postgresql.pkg.tar.zst" 2>/dev/null || echo 0)
    if [ "$fsize" -lt 1000000 ]; then
        echo "ERROR: Download too small ($fsize bytes). Check the URL."
        exit 1
    fi

    cd "$PG_SERVER_DIR"
    tar --use-compress-program=unzstd -xf postgresql.pkg.tar.zst
    echo "Extracted PostgreSQL server binaries to $PG_SERVER_DIR/usr/bin/"
    cd "$PROJECT_DIR"
}

_ensure_pg_server_binaries

# Add extracted PG binaries and libs to environment
if [ -d "$PG_SERVER_DIR/usr/bin" ]; then
    export PATH="$PG_SERVER_DIR/usr/bin:$PATH"
fi
if [ -d "$PG_SERVER_DIR/usr/lib" ]; then
    export LD_LIBRARY_PATH="${PG_SERVER_DIR}/usr/lib:${LD_LIBRARY_PATH:-}"
fi

echo "==================================================================="
echo " VecAdvisor++ Remote Setup"
echo " REMOTE_BASE : $REMOTE_BASE"
echo " PGDATA      : $PGDATA"
echo " PGPORT      : $PGPORT"
echo " PROJECT_DIR : $PROJECT_DIR"
echo " initdb      : $(which initdb)"
echo " pg_ctl      : $(which pg_ctl)"
echo "==================================================================="

# ---------------------------------------------------------------------------
# Step 1: User-level PostgreSQL cluster
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: PostgreSQL cluster ---"

if [ -d "$PGDATA" ] && [ -f "$PGDATA/PG_VERSION" ]; then
    echo "Cluster already exists at $PGDATA — skipping initdb."
else
    initdb -D "$PGDATA" --encoding=UTF8 --locale=C
    echo "Cluster initialised at $PGDATA."
fi

# Patch postgresql.conf (idempotent — uses sed to replace or append)
patch_conf() {
    local key="$1" value="$2"
    local conf="$PGDATA/postgresql.conf"
    if grep -q "^${key}" "$conf" 2>/dev/null; then
        sed -i "s|^${key}.*|${key} = ${value}|" "$conf"
    else
        echo "${key} = ${value}" >> "$conf"
    fi
}

patch_conf "port"                       "$PGPORT"
patch_conf "listen_addresses"           "'localhost'"
patch_conf "unix_socket_directories"    "'$PGDATA'"
patch_conf "shared_buffers"             "32GB"
patch_conf "effective_cache_size"       "200GB"
patch_conf "maintenance_work_mem"       "4GB"
patch_conf "work_mem"                   "256MB"
patch_conf "max_wal_size"               "4GB"
patch_conf "checkpoint_completion_target" "0.9"
patch_conf "max_connections"            "50"

echo "postgresql.conf patched."

# Start cluster (or do nothing if already running)
if pg_ctl -D "$PGDATA" status > /dev/null 2>&1; then
    echo "PostgreSQL already running."
else
    pg_ctl -D "$PGDATA" -l "$REMOTE_BASE/pg.log" start -w -t 60
    echo "PostgreSQL started. Logs: $REMOTE_BASE/pg.log"
fi

# Create database (idempotent)
createdb -h "$PGDATA" -p "$PGPORT" "$PGDB" 2>/dev/null \
    && echo "Database '$PGDB' created." \
    || echo "Database '$PGDB' already exists."

# ---------------------------------------------------------------------------
# Step 2: pgvector from source
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: pgvector ---"

PGVECTOR_DIR=$REMOTE_BASE/pgvector

# We need a pg_config wrapper because the system pg_config reports paths
# that don't exist (server headers are in our extracted package, not in
# /usr/include/postgresql/server). The wrapper redirects path queries.
PG_CONFIG_WRAPPER=$PG_SERVER_DIR/pg_config_wrapper.sh

cat > "$PG_CONFIG_WRAPPER" <<'WRAPPER'
#!/usr/bin/env bash
PG_BASE=/tmp2/b11902156/pg_server/usr
case "$1" in
    --pgxs)              echo "$PG_BASE/lib/postgresql/pgxs/src/makefiles/pgxs.mk" ;;
    --includedir-server) echo "$PG_BASE/include/postgresql/server" ;;
    --includedir)        echo "$PG_BASE/include" ;;
    --pkgincludedir)     echo "$PG_BASE/include/postgresql" ;;
    --pkglibdir)         echo "$PG_BASE/lib/postgresql" ;;
    --sharedir)          echo "$PG_BASE/share/postgresql" ;;
    --libdir)            echo "$PG_BASE/lib" ;;
    --bindir)            echo "$PG_BASE/bin" ;;
    --docdir)            echo "$PG_BASE/share/doc/postgresql" ;;
    --htmldir)           echo "$PG_BASE/share/doc/postgresql" ;;
    --localedir)         echo "$PG_BASE/share/locale" ;;
    --mandir)            echo "$PG_BASE/share/man" ;;
    --sysconfdir)        echo "$PG_BASE/etc/postgresql" ;;
    *)                   /usr/bin/pg_config "$@" ;;
esac
WRAPPER
chmod +x "$PG_CONFIG_WRAPPER"

if [ ! -d "$PGVECTOR_DIR" ]; then
    # Use main branch (v0.8.0 has API incompatibility with PG 18)
    git clone https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR"
fi

cd "$PGVECTOR_DIR"
make PG_CONFIG="$PG_CONFIG_WRAPPER" 2>&1
make install PG_CONFIG="$PG_CONFIG_WRAPPER" 2>&1
echo "pgvector compiled and installed."

cd "$PROJECT_DIR"

# Tell PostgreSQL where to find vector.so
patch_conf "dynamic_library_path" "'$PG_SERVER_DIR/usr/lib/postgresql:\$libdir'"

# Restart to pick up dynamic_library_path
pg_ctl -D "$PGDATA" restart -w -t 60 -l "$REMOTE_BASE/pg.log"

psql -h localhost -p "$PGPORT" "$PGDB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" \
    && echo "pgvector extension enabled." \
    || echo "WARNING: Could not create pgvector extension."

# ---------------------------------------------------------------------------
# Step 3: Rust toolchain + vecadvisor_rs
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Rust + vecadvisor_rs ---"

# uv (Python package manager)
if ! command -v uv &>/dev/null; then
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# Rust
if ! command -v cargo &>/dev/null; then
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
    source "$HOME/.cargo/env"
    echo "Rust installed."
else
    echo "Rust already installed: $(rustc --version)"
fi

source "$PROJECT_DIR/.venv/bin/activate"
uv pip install maturin --quiet

maturin develop --release --manifest-path "$PROJECT_DIR/vecadvisor_rs/Cargo.toml"
echo "vecadvisor_rs built and installed."

# Verify
python -c "import vecadvisor_rs; fns=dir(vecadvisor_rs); print('vecadvisor_rs functions:', [f for f in fns if not f.startswith('_')])"

# ---------------------------------------------------------------------------
# Step 4: Download datasets
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Datasets ---"

mkdir -p "$DATA_DIR/sift1m" "$DATA_DIR/gist1m"

# SIFT1M (~500 MB) — via Python loader
python - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from src.data.loader import download_sift1m
download_sift1m("/tmp2/b11902156/data/sift1m")
PYEOF

# GIST1M — ~3.6 GB
GIST_DIR=$DATA_DIR/gist1m
if [ -f "$GIST_DIR/gist/gist_base.fvecs" ]; then
    echo "GIST1M already present at $GIST_DIR/gist/"
else
    echo "Downloading GIST1M (~3.6 GB)..."
    wget --progress=dot:giga \
        ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz \
        -O "$GIST_DIR/gist.tar.gz"
    tar -xf "$GIST_DIR/gist.tar.gz" -C "$GIST_DIR/"
    echo "GIST1M extracted to $GIST_DIR/gist/"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==================================================================="
echo " Setup complete!"
echo ""
echo " PostgreSQL : localhost:$PGPORT  db=$PGDB"
echo " PGDATA     : $PGDATA"
echo " Data       : $DATA_DIR"
echo ""
echo " Before running experiments, set your environment:"
echo ""
echo "   export PATH=$PG_SERVER_DIR/usr/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=$PG_SERVER_DIR/usr/lib:\$LD_LIBRARY_PATH"
echo "   export PGDATA=$PGDATA"
echo "   source $PROJECT_DIR/.venv/bin/activate"
echo "   source \$HOME/.cargo/env"
echo "   source \$HOME/.local/bin/env"
echo ""
echo " Smoke test:"
echo "   python scripts/run_benchmark.py --config config/remote.yaml \\"
echo "       --n-base 10000 --n-queries 100 --k 10"
echo "==================================================================="

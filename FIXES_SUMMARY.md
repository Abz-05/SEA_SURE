# SEA_SURE Application Error Fixes

## Issues Identified

### 1. Missing Function: `create_fisher_map()`

**Error:** `vars() argument must have __dict__ attribute`
**Location:** Line 2350 in `app_integreted.py`
**Cause:** Function `create_fisher_map(fisher_name, my_catches)` is called but not defined

### 2. Database Schema Error with np.float64

**Error:** `schema "np" does not exist`
**Location:** `save_catch_to_db()` function
**Cause:** NumPy float64 types being passed to PostgreSQL, which interprets "np.float64" as a schema name

### 3. Deprecated Parameter Warnings

**Error:** `use_container_width` will be removed after 2025-12-31
**Location:** Multiple locations throughout the file
**Cause:** Streamlit API change - need to replace with `width` parameter

## Fixes Applied

### Fix 1: Add Missing `create_fisher_map()` Function

- Create a pydeck-based map visualization for fisher catches
- Use color coding based on freshness

### Fix 2: Convert NumPy Types to Native Python Types

- Ensure all float values from ML models are converted to Python floats
- Add type conversion in `save_catch_to_db()` function

### Fix 3: Replace Deprecated Parameters

- Replace all `width="stretch"` with `width="stretch"`
- Replace all `width="content"` with `width="content"`

"""
CSS styles for the Streamlit application.

This module contains all CSS styling for the SEA_SURE application.
"""

def get_styles() -> str:
    """
    Get CSS styles for the application.
    
    Returns:
        CSS string
    """
    return """
<style>
    /* Main Layout */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .fish-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.12);
        margin: 1rem 0;
        border: 2px solid #e3e6ea;
        transition: all 0.3s;
    }
    
    .fish-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
    }
    
    /* User Info */
    .user-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s;
    }
    
    .stats-card:hover {
        transform: scale(1.05);
    }
    
    .stats-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .stats-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Freshness Indicators */
    .freshness-high {
        color: #28a745;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .freshness-medium {
        color: #ffc107;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .freshness-low {
        color: #dc3545;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Status Badges */
    .status-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .status-available {
        background: #d4edda;
        color: #155724;
    }
    
    .status-sold {
        background: #f8d7da;
        color: #721c24;
    }
    
    .status-pending {
        background: #fff3cd;
        color: #856404;
    }
    
    .status-approved {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    .status-delivered {
        background: #d4edda;
        color: #155724;
    }
    
    /* Info Boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #0d47a1;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e65100;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1b5e20;
    }
</style>
"""

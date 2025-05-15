// API configuration for PlateVision app

// Base API URL (in development and production)
const BASE_URL = import.meta.env.VITE_API_BASE_URL;

// Export API endpoints
export const API_ENDPOINTS = {
  BASE_URL: BASE_URL,
  DETECT_LICENSE_PLATE: `${BASE_URL}/api/detect_license_plate/`,
  PROCESS_VIDEO: `${BASE_URL}/api/process_video/`,
  HEALTH_CHECK: `${BASE_URL}/api/health`
};

export default API_ENDPOINTS;
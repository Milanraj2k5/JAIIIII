const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`
    const token = localStorage.getItem('token')

    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    }

    const response = await fetch(url, config)

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  // Auth endpoints
  async signup(email, password) {
    const formData = new FormData()
    formData.append('email', email)
    formData.append('password', password)
    
    return this.request('/auth/signup', {
      method: 'POST',
      headers: {},
      body: formData,
    })
  }

  async login(email, password) {
    const formData = new FormData()
    formData.append('email', email)
    formData.append('password', password)
    
    return this.request('/auth/login', {
      method: 'POST',
      headers: {},
      body: formData,
    })
  }

  async getProfile() {
    return this.request('/auth/me')
  }

  // Upload endpoint
  async uploadFile(file, metadata = '') {
    const formData = new FormData()
    formData.append('file', file)
    if (metadata) {
      formData.append('metadata', metadata)
    }

    return this.request('/upload', {
      method: 'POST',
      headers: {},
      body: formData,
    })
  }

  // Results endpoints
  async getResult(resultId) {
    return this.request(`/results/${resultId}`)
  }

  async getHistory(skip = 0, limit = 20) {
    return this.request(`/history?skip=${skip}&limit=${limit}`)
  }

  // Blockchain endpoint
  async verifyOnBlockchain(resultId) {
    return this.request(`/blockchain/verify/${resultId}`, {
      method: 'POST',
    })
  }
}

export default new ApiService()
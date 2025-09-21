import { useState } from 'react'
import { useAuth } from '../utils/auth'
import { useNavigate } from 'react-router-dom'
import { Upload, AlertCircle } from 'lucide-react'

const UploadForm = () => {
  const [file, setFile] = useState(null)
  const [metadata, setMetadata] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { user } = useAuth()
  const navigate = useNavigate()

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      // Validate file type
      const allowedTypes = ['image/jpeg', 'image/png', 'video/mp4', 'audio/mp3', 'audio/wav']
      if (!allowedTypes.includes(selectedFile.type)) {
        setError('Unsupported file type. Please upload JPG, PNG, MP4, MP3, or WAV files.')
        return
      }

      // Validate file size
      const maxSize = selectedFile.type.startsWith('image/') ? 6 * 1024 * 1024 : 
                     selectedFile.type.startsWith('video/') ? 50 * 1024 * 1024 : 
                     20 * 1024 * 1024
      
      if (selectedFile.size > maxSize) {
        setError(`File too large. Max size: ${maxSize / (1024 * 1024)}MB`)
        return
      }

      setFile(selectedFile)
      setError('')
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return

    setError('')
    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)
      if (metadata) {
        formData.append('metadata', metadata)
      }

      const response = await fetch('/api/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const result = await response.json()
      navigate(`/result/${result.result_id}`)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (!user) {
    return (
      <div className="text-center">
        <p className="text-gray-600">Please log in to upload files.</p>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Media for Verification</h2>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-center space-x-2">
              <AlertCircle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          )}

          <div>
            <label htmlFor="file" className="block text-sm font-medium text-gray-700 mb-2">
              Select File
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors">
              <input
                type="file"
                id="file"
                onChange={handleFileChange}
                accept="image/*,video/*,audio/*"
                className="hidden"
              />
              <label htmlFor="file" className="cursor-pointer">
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">
                  {file ? file.name : 'Click to select a file'}
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Images: JPG, PNG (max 6MB) | Videos: MP4 (max 50MB) | Audio: MP3, WAV (max 20MB)
                </p>
              </label>
            </div>
          </div>

          <div>
            <label htmlFor="metadata" className="block text-sm font-medium text-gray-700">
              Additional Information (Optional)
            </label>
            <textarea
              id="metadata"
              value={metadata}
              onChange={(e) => setMetadata(e.target.value)}
              className="input-field mt-1"
              rows={3}
              placeholder="Describe the content or provide context..."
            />
          </div>

          <button
            type="submit"
            disabled={!file || loading}
            className="w-full btn-primary disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Upload and Analyze'}
          </button>
        </form>
      </div>
    </div>
  )
}

export default UploadForm
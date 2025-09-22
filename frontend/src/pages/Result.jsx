import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, ExternalLink, CheckCircle, XCircle, AlertTriangle } from 'lucide-react'
import TrustScoreChart from '../components/TrustScoreChart'
import { useAuth } from '../utils/auth'
import api from '../services/api'

const Result = () => {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!user) {
      navigate('/login')
      return
    }

    const fetchResult = async () => {
      try {
        const data = await api.getResult(id)
        setResult(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    fetchResult()
  }, [id, user, navigate])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">Error</h3>
        <p className="text-gray-600 mb-4">{error || 'Result not found'}</p>
        <button onClick={() => navigate('/')} className="btn-primary">
          Go Home
        </button>
      </div>
    )
  }

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case 'Real':
        return <CheckCircle className="w-8 h-8 text-green-500" />
      case 'Fake':
        return <XCircle className="w-8 h-8 text-red-500" />
      default:
        return <AlertTriangle className="w-8 h-8 text-yellow-500" />
    }
  }

  const getVerdictColor = (verdict) => {
    switch (verdict) {
      case 'Real':
        return 'bg-green-100 text-green-800'
      case 'Fake':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-yellow-100 text-yellow-800'
    }
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <button
          onClick={() => navigate('/history')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to History</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Main Result */}
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-4">
                {getVerdictIcon(result.verdict)}
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">{result.file_name}</h1>
                  <p className="text-gray-600">{result.file_type.toUpperCase()} • {new Date(result.created_at).toLocaleDateString()}</p>
                </div>
              </div>
              <span className={`px-4 py-2 rounded-full text-lg font-medium ${getVerdictColor(result.verdict)}`}>
                {result.verdict}
              </span>
            </div>

            <div className="mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-lg font-medium text-gray-700">Trust Score</span>
                <span className="text-4xl font-bold text-gray-900">{result.trust_score}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full ${
                    result.trust_score >= 70 ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${result.trust_score}%` }}
                ></div>
              </div>
            </div>

            {result.analysis.onchain_tx && (
              <div className="flex items-center space-x-2 text-blue-600">
                <ExternalLink className="w-5 h-5" />
                <a 
                  href={`https://mumbai.polygonscan.com/tx/${result.analysis.onchain_tx}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline font-medium"
                >
                  View on Blockchain
                </a>
              </div>
            )}
          </div>

          {/* Analysis Details */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Details</h3>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-700 mb-2">AI Analysis</h4>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-sm text-gray-600">Score: {(result.analysis.ai_analysis?.score * 100 || 0).toFixed(1)}%</p>
                  <p className="text-sm text-gray-600">Model: {result.analysis.ai_analysis?.model || 'N/A'}</p>
                  {result.analysis.ai_analysis?.explanations && (
                    <ul className="mt-2 text-sm text-gray-600">
                      {result.analysis.ai_analysis.explanations.map((explanation, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <span className="text-gray-400">•</span>
                          <span>{explanation}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-2">Azure Analysis</h4>
                <div className="bg-gray-50 p-3 rounded-lg">
                  {result.analysis.azure_checked ? (
                    <>
                      <p className="text-sm text-gray-600">Score: {(result.analysis.azure_analysis?.score * 100 || 0).toFixed(1)}%</p>
                      <p className="text-sm text-gray-600">Service: {result.analysis.azure_analysis?.service || 'N/A'}</p>
                    </>
                  ) : (
                    <p className="text-sm text-gray-500">Azure analysis not available</p>
                  )}
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-2">News Verification</h4>
                <div className="bg-gray-50 p-3 rounded-lg">
                  {result.analysis.news_checked ? (
                    <>
                      <p className="text-sm text-gray-600">Found: {result.analysis.news_analysis?.found ? 'Yes' : 'No'}</p>
                      <p className="text-sm text-gray-600">Articles: {result.analysis.news_analysis?.article_count || 0}</p>
                      {result.analysis.news_analysis?.articles && result.analysis.news_analysis.articles.length > 0 && (
                        <div className="mt-2">
                          <p className="text-sm font-medium text-gray-700">Related Articles:</p>
                          <ul className="mt-1 space-y-1">
                            {result.analysis.news_analysis.articles.slice(0, 3).map((article, index) => (
                              <li key={index}>
                                <a 
                                  href={article.url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-sm text-blue-600 hover:underline"
                                >
                                  {article.title}
                                </a>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  ) : (
                    <p className="text-sm text-gray-500">News verification not available</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div>
          <TrustScoreChart analysis={result.analysis} />
        </div>
      </div>
    </div>
  )
}

export default Result
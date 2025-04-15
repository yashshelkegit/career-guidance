import React, { useState } from 'react';

const CareerAssessment = () => {
  const [activeTab, setActiveTab] = useState('skills');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({
    media: null,
    lawGov: null,
    tech: null,
    aptitude: null,
    gmm: null
  });

  // Pre-filled skills data
  const [skillsData, setSkillsData] = useState({
    Social_Studies_Score: 85,
    Language_Score: 90,
    Critical_Thinking: 8,
    Debate_Skills: 7,
    Public_Speaking: 9,
    Leadership: 8,
    Research_Skills: 7,
    Communication_Skills: 9,
    Time_Management: 8,
    Adaptability: 9
  });

  // Pre-filled tech data
  const [techData, setTechData] = useState({
    Math_Score: 90,
    Science_Score: 65,
    Logical_Thinking: 3,
    Problem_Solving_Skills: 1,
    Programming_Knowledge: 3,
    Coding_Experience: 4,
    Tech_Exposure: 3,
    Interest_in_Tech: 3,
    Participation_in_Coding_Clubs: 6,
    Participation_in_Science_Fairs: 5,
    STEM_Activities: 8,
    Attention_to_Detail: 5,
    Project_Work_Experience: 15,
    Communication_Skills: 2,
    Time_Management: 3,
    Adaptability: 5,
    Digital_Literacy: 2
  });

  // Pre-filled aptitude and personality data
  const [gmmData, setGmmData] = useState({
    O_score: 8.78,
    C_score: 7.45,
    E_score: 5.34,
    A_score: 8.45,
    N_score: 6.01,
    Numerical_Aptitude: 4.56,
    Spatial_Aptitude: 4.01,
    Perceptual_Aptitude: 4.23,
    Abstract_Reasoning: 6.67,
    Verbal_Reasoning: 9.23
  });

  const [aptitudeData, setAptitudeData] = useState({
    Numerical_Aptitude: 2.0,
    Spatial_Aptitude: 1.8,
    Perceptual_Aptitude: 2.2,
    Abstract_Reasoning: 2.0,
    Verbal_Reasoning: 2.5
  });

  const handleSkillsChange = (e) => {
    setSkillsData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleTechChange = (e) => {
    setTechData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleGmmChange = (e) => {
    setGmmData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleAptitudeChange = (e) => {
    setAptitudeData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  // Existing predict functions...
  const predictMedia = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8002/predict/media', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(skillsData)
      });
      const data = await response.json();
      setResults(prev => ({ ...prev, media: data }));
    } catch (error) {
      console.error('Error predicting media:', error);
    } finally {
      setLoading(false);
    }
  };

  const predictLawGov = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8002/predict/law-gov', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(skillsData)
      });
      const data = await response.json();
      setResults(prev => ({ ...prev, lawGov: data }));
    } catch (error) {
      console.error('Error predicting law/gov:', error);
    } finally {
      setLoading(false);
    }
  };

  const predictTech = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8002/predict/tech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(techData)
      });
      const data = await response.json();
      setResults(prev => ({ ...prev, tech: data }));
    } catch (error) {
      console.error('Error predicting tech:', error);
    } finally {
      setLoading(false);
    }
  };

  const predictAptitude = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://0.0.0.0:8002/predict/aptitude', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(aptitudeData)
      });
      const data = await response.json();
      setResults(prev => ({ ...prev, aptitude: data }));
    } catch (error) {
      console.error('Error predicting aptitude:', error);
    } finally {
      setLoading(false);
    }
  };

  const predictGmm = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8002/predict/gmm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gmmData)
      });
      const data = await response.json();
      setResults(prev => ({ ...prev, gmm: data }));
    } catch (error) {
      console.error('Error predicting GMM:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Career Assessment</h1>
        
        {/* Tab Navigation */}
        <div className="flex mb-6 border-b">
          <button
            className={`px-4 py-2 font-medium ${activeTab === 'skills' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('skills')}
          >
            Skills Assessment
          </button>
          <button
            className={`px-4 py-2 font-medium ${activeTab === 'tech' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('tech')}
          >
            Tech Assessment
          </button>
          <button
            className={`px-4 py-2 font-medium ${activeTab === 'aptitude' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('aptitude')}
          >
            Aptitude Test
          </button>
          <button
            className={`px-4 py-2 font-medium ${activeTab === 'gmm' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
            onClick={() => setActiveTab('gmm')}
          >
            Personality & Aptitude
          </button>
        </div>

        {/* Skills Assessment Form */}
        {activeTab === 'skills' && (
          <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
            <h2 className="text-xl font-semibold mb-4">Skills Assessment</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.keys(skillsData).map((field) => (
                <div key={field}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {field.replace(/_/g, ' ')}
                  </label>
                  <input
                    type="number"
                    name={field}
                    value={skillsData[field]}
                    onChange={handleSkillsChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    min="0"
                    max={field.includes('Score') ? 100 : 10}
                    step={field.includes('Score') ? 1 : 0.1}
                  />
                </div>
              ))}
            </div>
            <div className="mt-6 flex gap-4">
              <button
                onClick={predictMedia}
                className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={loading}
              >
                Predict Media Career
              </button>
              <button
                onClick={predictLawGov}
                className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
                disabled={loading}
              >
                Predict Law/Government Career
              </button>
            </div>
          </div>
        )}

        {/* Tech Assessment Form */}
        {activeTab === 'tech' && (
          <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
            <h2 className="text-xl font-semibold mb-4">Tech Skills Assessment</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.keys(techData).map((field) => (
                <div key={field}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {field.replace(/_/g, ' ')}
                  </label>
                  <input
                    type="number"
                    name={field}
                    value={techData[field]}
                    onChange={handleTechChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    min="0"
                    max={field.includes('Score') ? 100 : field === 'Project_Work_Experience' ? 20 : 10}
                    step={field.includes('Score') ? 1 : 0.1}
                  />
                </div>
              ))}
            </div>
            <div className="mt-6">
              <button
                onClick={predictTech}
                className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                disabled={loading}
              >
                Predict Tech Career
              </button>
            </div>
          </div>
        )}

        {/* Aptitude Test Form */}
        {activeTab === 'aptitude' && (
          <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
            <h2 className="text-xl font-semibold mb-4">Aptitude Test</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.keys(aptitudeData).map((field) => (
                <div key={field}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {field.replace(/_/g, ' ')}
                  </label>
                  <input
                    type="number"
                    name={field}
                    value={aptitudeData[field]}
                    onChange={handleAptitudeChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    step="0.1"
                    min="0"
                    max="5"
                  />
                </div>
              ))}
            </div>
            <div className="mt-6">
              <button
                onClick={predictAptitude}
                className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2"
                disabled={loading}
              >
                Predict Aptitude
              </button>
            </div>
          </div>
        )}

        {/* GMM Assessment Form */}
        {activeTab === 'gmm' && (
          <div className="bg-white p-6 rounded-lg shadow-sm mb-6">
            <h2 className="text-xl font-semibold mb-4">Personality & Aptitude Assessment</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.keys(gmmData).map((field) => (
                <div key={field}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {field.replace(/_/g, ' ')}
                  </label>
                  <input
                    type="number"
                    name={field}
                    value={gmmData[field]}
                    onChange={handleGmmChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
                    step="0.01"
                    min="0"
                    max="10"
                  />
                </div>
              ))}
            </div>
            <div className="mt-6">
              <button
                onClick={predictGmm}
                className="bg-teal-600 text-white px-4 py-2 rounded-md hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2"
                disabled={loading}
              >
                Predict Performance Cluster
              </button>
            </div>
          </div>
        )}

        {/* Results Display */}
        {loading && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          </div>
        )}

                {(results.media || results.lawGov || results.tech || results.aptitude || results.gmm) && (
          <div className="bg-white p-6 rounded-lg shadow-sm mt-6">
            <h2 className="text-xl font-semibold mb-4">Results</h2>
            {results.media && (
              <div className="mb-4">
                <h3 className="text-lg font-medium text-gray-700">Media & Communication:</h3>
                <p className="text-gray-600">
                  Suitability Score: <span className="font-semibold">{results.media.media_communication}</span>
                </p>
              </div>
            )}
            {results.lawGov && (
              <div className="mb-4">
                <h3 className="text-lg font-medium text-gray-700">Law & Government:</h3>
                <p className="text-gray-600">
                  Suitability Score: <span className="font-semibold">{results.lawGov.law_and_government}</span>
                </p>
              </div>
            )}
            {results.tech && (
              <div className="mb-4">
                <h3 className="text-lg font-medium text-gray-700">Tech Career:</h3>
                <p className="text-gray-600">
                  Suitability Score: <span className="font-semibold">{results.tech.suitability_score}</span>
                </p>
              </div>
            )}
            {results.aptitude && (
              <div className="mb-4">
                <h3 className="text-lg font-medium text-gray-700">Aptitude Test:</h3>
                <p className="text-gray-600">
                  Competency Level: <span className="font-semibold">{results.aptitude.competency_level}</span>
                </p>
              </div>
            )}
            {results.gmm && (
              <div className="mb-4">
                <h3 className="text-lg font-medium text-gray-700">Personality & Aptitude:</h3>
                <p className="text-gray-600">
                  Cluster: <span className="font-semibold">{results.gmm.cluster}</span>
                </p>
                <p className="text-gray-600">
                  Competency Level: <span className="font-semibold">{results.gmm.competency_level}</span>
                </p>
                <p className="text-gray-600">
                  Probabilities: <span className="font-semibold">{JSON.stringify(results.gmm.probabilities)}</span>
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CareerAssessment;
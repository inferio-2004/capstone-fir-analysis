import React from 'react';
import { GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';
import './Login.css';

const LoginPage = ({ onLoginSuccess }) => {
  const handleSuccess = (credentialResponse) => {
    const decoded = jwtDecode(credentialResponse.credential);
    console.log('Google Auth Decoded:', decoded);
    const userData = {
      token: credentialResponse.credential,
      name: decoded.name,
      email: decoded.email,
      picture: decoded.picture,
    };
    onLoginSuccess(userData);
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-logo">⚖</div>
        <h1>LexIR</h1>
        <p>Legal Intelligence & Retrieval</p>
        <div className="google-login-btn">
          <GoogleLogin
            onSuccess={handleSuccess}
            onError={() => {
              console.log('Login Failed');
              alert('Login Failed. Please try again.');
            }}
            useOneTap
          />
        </div>
      </div>
    </div>
  );
};

export default LoginPage;

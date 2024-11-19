import React, { useState } from "react";
import { useForm } from "react-hook-form";
import axios from "axios";
import "./App.css"; // Import the external CSS

function App() {
  const { register, handleSubmit, reset } = useForm();
  const [sentiment, setSentiment] = useState("");
  const [error, setError] = useState("");
  const [submittedReview, setSubmittedReview] = useState(""); 

  const onSubmit = async (data) => {
    setError("");
    setSentiment("");
    setSubmittedReview(data.review);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        review: data.review,
      });
      setSentiment(response.data.sentiment);
      reset(); // Reset the form after submission
    } catch (err) {
      setError("An error occurred. Please try again.");
      console.error(err);
    }
  };

  return (
    <>
      <div className="main1">
        <div className="app-container">
          <h1 className="title">Sentiment Analysis</h1>
          <form onSubmit={handleSubmit(onSubmit)} className="form">
            <textarea
              {...register("review", { required: "Please enter a review." })}
              placeholder="Enter a product review..."
              rows="5"
              className="textarea"
            />
            <button type="submit" className="button">
              Analyze Sentiment
            </button>
          </form>

          {sentiment && (
            <>
              <div className="question">{submittedReview && (<p>{submittedReview}</p>)}</div>
              <p
                className={`result ${
                  sentiment === "Negative" ? "error" : "success"
                }`}
              >
                Sentiment: <strong>{sentiment}</strong>
              </p>
            </>
          )}

          {error && <p className="result error">{error}</p>}
        </div>
      </div>
    </>
  );
}

export default App;

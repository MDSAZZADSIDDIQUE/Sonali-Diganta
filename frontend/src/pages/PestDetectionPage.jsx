import { useRef, useState } from "react";
import Navbar from "../components/Navbar";

export default function PestDetectionPage() {
  const imageRef = useRef();
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [imageURL, setImageURL] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    const url = URL.createObjectURL(selectedFile);
    setImageURL(url);
    setFile(selectedFile);
  };

  const handleSubmit = async (event) => {
    setLoading(true);
    event.preventDefault();
    const formData = new FormData();
    formData.append("image", file);
    try {
      const response = await fetch("http://127.0.0.1:5000/pest-detection", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction(data);
      setLoading(false);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <>
      <Navbar />
      <div className="flex justify-center">
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body flex justify-center items-center space-y-3">
            <h2 className="card-title">Pest Detection</h2>
            <form
              onSubmit={handleSubmit}
              className="flex justify-center items-center flex-col space-y-6"
            >
              <input
                type="file"
                className="file-input file-input-bordered file-input-primary w-full max-w-xs"
                onChange={handleFileChange}
              />
              <figure>
                {imageURL && <img src={imageURL} ref={imageRef} />}
              </figure>
              <button className="btn btn-primary" type="submit">
                Predict
              </button>
            </form>

            {loading && (
              <span className="loading loading-spinner text-primary loading-lg"></span>
            )}

            {prediction && (
              <>
                <h2 className="card-title">{prediction.class}</h2>
                <p>{prediction.confidence}%</p>
                <div className="card-actions justify-end">
                  <button className="btn btn-primary">Learn More</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

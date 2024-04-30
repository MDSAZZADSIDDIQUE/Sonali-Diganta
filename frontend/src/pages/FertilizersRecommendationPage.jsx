import { useState } from "react";
import Navbar from "../components/Navbar";

export default function FertilizersRecommendationPage() {
  const [temperature, setTemperature] = useState(null);
  const [humidity, setHumidity] = useState(null);
  const [moisture, setMoisture] = useState(null);
  const [soilType, setSoilType] = useState(null);
  const [cropType, setCropType] = useState(null);
  const [nitrogen, setNitrogen] = useState(null);
  const [potassium, setPotassium] = useState(null);
  const [phosphorous, setPhosphorous] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    setLoading(true);
    event.preventDefault();
    const formData = new FormData();
    formData.append("Temparature", temperature);
    formData.append("Humidity", humidity);
    formData.append("Moisture", moisture);
    formData.append("Nitrogen", nitrogen);
    formData.append("Potassium", potassium);
    formData.append("Phosphorous", phosphorous);
    try {
      const response = await fetch(
        "http://127.0.0.1:5000/fertilizerrecommendation",
        {
          method: "POST",
          body: formData,
        }
      );
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
            <h2 className="card-title">Fertilizer Recommendation</h2>
            <form
              onSubmit={handleSubmit}
              className="flex justify-center items-center flex-col space-y-6"
            >
              <label className="input input-bordered flex items-center gap-2 w-full">
                Temparature
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="n"
                  onChange={(event) => {
                    setTemperature(event.target.value);
                  }}
                />
              </label>
              <label className="input input-bordered flex items-center gap-2 w-full">
                Humidity
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="p"
                  onChange={(event) => {
                    setHumidity(event.target.value);
                  }}
                />
              </label>
              <label className="input input-bordered flex items-center gap-2 w-full">
                Moisture
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="k"
                  onChange={(event) => {
                    setMoisture(event.target.value);
                  }}
                />
              </label>
              <select className="select select-bordered w-full max-w-xs">
                <option disabled selected>
                  Soil Type
                </option>
                <option>Black</option>
                <option>Clayey</option>
                <option>Loamy</option>
                <option>Red</option>
                <option>Sandy</option>
              </select>
              <label className="input input-bordered flex items-center gap-2 w-full">
                Crop Type
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="temperature"
                  onChange={(event) => {
                    setCropType(event.target.value);
                  }}
                />
              </label>
              <label className="input input-bordered flex items-center gap-2 w-full">
                Nitrogen
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="humidity"
                  onChange={(event) => {
                    setNitrogen(event.target.value);
                  }}
                />
              </label>
              <label className="input input-bordered flex items-center gap-2 w-full">
                Potassium
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="ph"
                  onChange={(event) => {
                    setPotassium(event.target.value);
                  }}
                />
              </label>
              <label className="input input-bordered flex items-center gap-2 w-full">
                Phosphorous
                <input
                  type="text"
                  className="grow"
                  placeholder="Daisy"
                  name="rainfall"
                  onChange={(event) => {
                    setPhosphorous(event.target.value);
                  }}
                />
              </label>

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
                {/* <p>{prediction.confidence}%</p> */}
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

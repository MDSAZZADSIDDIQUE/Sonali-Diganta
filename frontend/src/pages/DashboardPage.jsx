import Box from "@mui/material/Box";
import { useNavigate } from "react-router-dom";
import Footer from "../components/Footer";
import Navbar from "../components/Navbar";

const DashboardPage = () => {
  const navigate = useNavigate();
  const redirectToPlantDiseaseDetection = () => {
    navigate("/plant-disease-detection");
  };
  const redirectToCropRecommendation = () => {
    navigate("/crop-recommendation");
  };
  const redirectToEnvironmentImpactOfFoodProduction = () => {
    navigate("/environment-impact-of-food-production");
  };
  const redirectToPlantGrowthStage = () => {
    navigate("/plant-growth-stage");
  };
  const redirectToCropYieldPrediction = () => {
    navigate("/crop-yield-prediction");
  };
  const redirectToFertilizersRecommendation = () => {
    navigate("/fertilizers-recommendation");
  };
  const redirectToPestDetection = () => {
    navigate("/pest-detection");
  };
  return (
    <Box>
      <div>
        <Navbar />
        <div
          className="hero min-h-screen"
          style={{
            backgroundImage: "url(wp1886357-agriculture-wallpapers.jpg)",
          }}
        >
          <div className="hero-overlay bg-opacity-60"></div>
          <div className="hero-content text-center text-neutral-content">
            <div className="max-w-md flex flex-col space-y-5">
              <button
                className="btn btn-primary"
                onClick={redirectToPlantDiseaseDetection}
              >
                Plant Disease Detection
              </button>
              <button
                className="btn btn-primary"
                onClick={redirectToCropRecommendation}
              >
                Crop Recommendation
              </button>
              <button
                className="btn btn-primary"
                onClick={redirectToEnvironmentImpactOfFoodProduction}
              >
                Environment Impact of Food Production
              </button>
              <button
                className="btn btn-primary"
                onClick={redirectToPlantGrowthStage}
              >
                Plant Growth Stage
              </button>
              <button
                className="btn btn-primary"
                onClick={redirectToCropYieldPrediction}
              >
                Crop Yeild Prediction
              </button>
              <button
                className="btn btn-primary"
                onClick={redirectToFertilizersRecommendation}
              >
                Fertilizer Recommendation
              </button>
              <button
                className="btn btn-primary"
                onClick={redirectToPestDetection}
              >
                Pest Detection
              </button>
            </div>
          </div>
        </div>
        <Footer />
      </div>
    </Box>
  );
};

export default DashboardPage;

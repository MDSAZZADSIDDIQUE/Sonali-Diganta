import React from "react";
import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from "react-router-dom";
import App from "./App";
import "./index.css";
import CropRecommendationPage from "./pages/CropRecommendationPage";
import CropYieldPredictionPage from "./pages/CropYieldPredictionPage";
import DashboardPage from "./pages/DashboardPage";
import EnvironmentImpactOfFoodProductionPage from "./pages/EnvironmentImpactOfFoodProductionPage";
import FertilizersRecommendationPage from "./pages/FertilizersRecommendationPage";
import PestDetectionPage from "./pages/PestDetectionPage";
import PlantDiseaseDetectionPage from "./pages/PlantDiseaseDetectionPage";
import PlantGrowthStagePage from "./pages/PlantGrowthStagePage";

const router = createBrowserRouter(
  createRoutesFromElements(
    <Route path="/">
      <Route path="/" element={<App />} />
      <Route path="dashboard" element={<DashboardPage />} />
      <Route
        path="plant-disease-detection"
        element={<PlantDiseaseDetectionPage />}
      />
      <Route path="crop-recommendation" element={<CropRecommendationPage />} />
      <Route
        path="environment-impact-of-food-production"
        element={<EnvironmentImpactOfFoodProductionPage />}
      />
      <Route path="plant-growth-stage" element={<PlantGrowthStagePage />} />
      <Route
        path="crop-yield-prediction"
        element={<CropYieldPredictionPage />}
      />
      <Route
        path="fertilizers-recommendation"
        element={<FertilizersRecommendationPage />}
      />
      <Route path="pest-detection" element={<PestDetectionPage />} />
    </Route>
  )
);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);

import Papa from "papaparse";
import React, { useEffect, useState } from "react";
export default function EnvironmentImpactOfFoodProductionPage() {
  const [hello, setHello] = useState(null);
  useEffect(() => {
    fetch("./Food_Production.csv")
      .then((response) => response.text())
      .then((responseText) => {
        var data = Papa.parse(responseText);
        setHello(data.data);
      });
  }, []);
  return <>{hello}</>;
}

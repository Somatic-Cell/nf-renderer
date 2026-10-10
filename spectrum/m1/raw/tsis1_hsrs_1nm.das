attributes {
    samples{
        wavelength{
            string units "nm";
            string precision "1";
            string long_name "Wavelength";
        }
        unknown{
            irradiance{
                string units "W/m^2/nm";
                string long_name "Solar Spectral Irradiance";
                string standard_name "solar_irradiance_per_unit_wavelength";
            }
            uncertainty{
                string units "W/m^2/nm";
                string long_name "Uncertainty in Solar Spectral Irradiance";
            }
            bandwidth{
                string units "nm";
                string precision "1";
                string long_name "Wavelength Band Width";
            }
        }
    }
}
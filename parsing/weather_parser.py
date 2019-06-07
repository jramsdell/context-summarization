
import numpy as np

class WeatherParser(object):
    def __init__(self, weather_loc):
        self.weather_loc = weather_loc


    def _zero_if_na(self, datum):
        if datum == 'NA':
            return 0.0
        return float(datum)

    def _yes_no_to_binary(self, datum):
        if datum == 'Yes':
            return 1
        return 0


    def parse(self):
        with open(self.weather_loc, 'r') as f:
            return self._parse(f)

    def _parse(self, f):
        results = []
        mappings = {}

        first = 1
        for line in f:
            elements = line.rstrip().split(",")
            if first:
                for idx, i in enumerate(elements):
                    mappings[idx] = i
                first = 0
                continue

            to_mapping = dict([(mappings[idx], i) for idx, i in enumerate(elements)])
            results.append(
                [
                    self._zero_if_na(to_mapping["MinTemp"]),
                    self._zero_if_na(to_mapping["MaxTemp"]),
                    self._zero_if_na(to_mapping["Rainfall"]),
                    self._zero_if_na(to_mapping["WindGustSpeed"]),
                    self._zero_if_na(to_mapping["WindSpeed9am"]),
                    self._zero_if_na(to_mapping["WindSpeed3pm"]),
                    self._zero_if_na(to_mapping["Humidity9am"]),
                    self._zero_if_na(to_mapping["Humidity3pm"]),
                    self._zero_if_na(to_mapping["Pressure9am"]),
                    self._zero_if_na(to_mapping["Pressure3pm"]),
                    self._zero_if_na(to_mapping["Cloud9am"]),
                    self._zero_if_na(to_mapping["Cloud3pm"]),
                    self._zero_if_na(to_mapping["Temp9am"]),
                    self._zero_if_na(to_mapping["Temp3pm"]),
                    self._yes_no_to_binary(to_mapping["RainToday"]),
                    self._yes_no_to_binary(to_mapping["RainTomorrow"])
                ]
            )
            # min_temp = float(elements[2])
            # results.append([mappings[idx] + ": " + i for idx, i in enumerate(elements)])

            # if len(results) > 30:
            #     break

        # for i in results:
        #     print(i)
        results = np.asarray(results)
        predictors = results.T[0:results.shape[1] - 1].T
        labels: np.ndarray = results.T[-1].T

        return predictors, labels


if __name__ == '__main__':
    wp = WeatherParser("/home/jsc57/jupyter/data/weather/weatherAUS.csv")
    wp.parse()
    # WeatherParser("")


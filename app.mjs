export const REQUIRED_FIELDS = ["resting_hr", "hrv", "sleep_hours", "training_load", "soreness"];

const FIELD_ALIASES = {
  resting_hr: ["resting_hr", "restinghr", "tetno_spoczynkowe", "restingheartrate"],
  hrv: ["hrv"],
  sleep_hours: ["sleep_hours", "sleep", "sen_godziny", "hours_sleep"],
  training_load: ["training_load", "load", "obciazenie_treningowe"],
  soreness: ["soreness", "bol_miesni", "rpe"],
  mood: ["mood", "samopoczucie"]
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-value));

const parseNumber = (value) => {
  const normalized = String(value ?? "").trim().replace(",", ".");
  if (!normalized) return Number.NaN;
  return Number(normalized);
};

const findHeaderKey = (headers, field) => {
  const aliases = FIELD_ALIASES[field] ?? [field];
  return aliases.find((alias) => headers.includes(alias));
};

export function parseMedicalCsv(csvText) {
  const lines = String(csvText)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("Plik CSV musi zawierać nagłówek i co najmniej jeden wiersz danych.");
  }

  const delimiter = lines[0].includes(";") ? ";" : ",";
  const headers = lines[0].split(delimiter).map((header) => header.trim().toLowerCase());
  const rows = lines.slice(1).map((line) => line.split(delimiter).map((value) => value.trim()));
  const latest = rows[rows.length - 1];

  const data = {};
  for (const field of Object.keys(FIELD_ALIASES)) {
    const headerKey = findHeaderKey(headers, field);
    if (!headerKey) continue;
    const index = headers.indexOf(headerKey);
    data[field] = parseNumber(latest[index]);
  }

  for (const field of REQUIRED_FIELDS) {
    if (!Number.isFinite(data[field])) {
      throw new Error(`Brak pola '${field}' lub niepoprawna wartość w pliku.`);
    }
  }

  if (!Number.isFinite(data.mood)) {
    data.mood = 5;
  }

  return data;
}

export function evaluateMetrics(data) {
  const restingHrNorm = clamp((data.resting_hr - 40) / 40, 0, 1);
  const hrvRiskNorm = clamp(1 - data.hrv / 120, 0, 1);
  const sleepDeficitNorm = clamp((8 - data.sleep_hours) / 4, 0, 1);
  const trainingLoadNorm = clamp(data.training_load / 100, 0, 1);
  const sorenessNorm = clamp(data.soreness / 10, 0, 1);
  const moodRiskNorm = clamp(1 - data.mood / 10, 0, 1);

  const riskScore =
    -1.2 +
    1.6 * restingHrNorm +
    1.8 * hrvRiskNorm +
    1.4 * sleepDeficitNorm +
    1.5 * trainingLoadNorm +
    1.3 * sorenessNorm +
    1.0 * moodRiskNorm;

  const overtrainingProbability = Math.round(sigmoid(riskScore) * 100);

  const recoveryPenalty =
    0.25 * sleepDeficitNorm +
    0.25 * hrvRiskNorm +
    0.2 * sorenessNorm +
    0.15 * trainingLoadNorm +
    0.15 * restingHrNorm;

  const regenerationLevel = Math.round(clamp((1 - recoveryPenalty) * 100, 0, 100));

  return { overtrainingProbability, regenerationLevel };
}

export function generateAdvice({ overtrainingProbability, regenerationLevel }) {
  const advice = [];

  if (overtrainingProbability >= 70) {
    advice.push("Wysokie ryzyko przetrenowania: ogranicz intensywność treningu na 24-72h.");
  } else if (overtrainingProbability >= 40) {
    advice.push("Umiarkowane ryzyko przetrenowania: zaplanuj lżejszą jednostkę treningową.");
  } else {
    advice.push("Niskie ryzyko przetrenowania: możesz kontynuować plan treningowy.");
  }

  if (regenerationLevel < 40) {
    advice.push("Niski poziom regeneracji: zwiększ sen i rozważ dzień pełnego odpoczynku.");
  } else if (regenerationLevel < 70) {
    advice.push("Średnia regeneracja: zadbaj o nawodnienie i regenerację aktywną.");
  } else {
    advice.push("Wysoka regeneracja: utrzymuj obecny poziom odpoczynku i żywienia.");
  }

  return advice;
}

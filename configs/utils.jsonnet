local bool(s) =
  if s == "true" then true
  else if s == "false" then false
  else error "invalid boolean: " + std.manifestJson(s);

local getTransformerDim(modelName) =
  if modelName == "bert-base-cased" then 768
  else if modelName == "bert-large-cased" then 1024
  else if modelName == "roberta-base" then 768
  else if modelName == "roberta-large" then 1024
  else if modelName == "t5-base" then 768
  else if modelName == "t5-large" then 1024
  else error "Invalid model: " + std.manifestJson(modelName);

{
    bool:: bool,
    getTransformerDim:: getTransformerDim,
}

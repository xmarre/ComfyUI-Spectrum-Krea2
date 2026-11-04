# Architecture

Implementation in progress. Native integration targets ComfyUI SingleStreamDiT, captures image-token features immediately before LastLayer, and always retains current-timestep native output projection. Sampling state is scoped to each SAMPLER_SAMPLE invocation; independent conditioning UUID sequences own separate histories. GPU quality and performance are not yet validated.

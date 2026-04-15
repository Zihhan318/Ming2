# Ming Architecture Simple

## 1. Overall Architecture

```mermaid
flowchart TB
    A[User Input] --> B[Processor]
    B --> B1[Text Tokens]
    B --> B2[Image Video Preprocess]
    B --> B3[Audio Preprocess]

    subgraph U[Unified Backbone]
        C1[Vision Encoder]
        C2[Audio Encoder]
        C3[Image Projector]
        C4[Audio Projector]
        C5[MoE LLM]
        C6[Text Output]
    end

    B2 --> C1 --> C3
    B3 --> C2 --> C4
    B1 --> C5
    C3 --> C5
    C4 --> C5
    C5 --> C6

    subgraph G[Optional Image Generation]
        G1[Generation Tokens]
        G2[Selected LLM States]
        G3[Connector]
        G4[ByT5 Optional]
        G5[Diffusion]
        G6[Image Output]
    end

    B1 -.-> G1
    B2 -.-> G1
    G1 -. append to LLM input .-> C5
    C5 -. select gen-token states .-> G2
    G2 --> G3
    G3 --> G5
    G4 -. extra text cond .-> G5
    G5 --> G6

    subgraph T[Optional Talker load_talker=True, independent of main MoE path]
        T1[Speaker Extractor]
        T2[BailingTalker2 Qwen2]
        T3[CFM DiT]
        T4[Aggregator Stop Head]
        T5[AudioVAE]
        T6[Waveform Output]
    end

    A -. prompt wav .-> T1
    A -. prompt text .-> T2
    T1 --> T2
    T2 --> T3 --> T4 --> T2
    T3 --> T5 --> T6
```

## 2. Understanding Flow

```mermaid
sequenceDiagram
    participant U as User
    participant P as Processor
    participant IP as ImageProcessor
    participant AP as AudioProcessor
    participant VE as VisionEncoder
    participant AE as AudioEncoder
    participant PR as Projector
    participant L as MoE_LLM
    participant O as Output

    U->>P: text image video audio
    P->>P: template plus tokenize text

    opt image or video
        P->>IP: process image video
        IP-->>P: pixel_values and grid
    end

    opt audio
        P->>AP: process audio
        AP-->>P: audio feats and lengths
    end

    P-->>L: input_ids and modality tensors

    opt image or video
        L->>VE: extract visual feature
        VE-->>PR: visual tokens
        PR-->>L: visual embeds
    end

    opt audio
        L->>AE: extract audio feature
        AE-->>PR: audio tokens
        PR-->>L: audio embeds
    end

    L->>L: prompt_wrap_navit and generate
    L-->>O: generated ids
    O-->>U: decoded text
```

## 3. Image Generation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant P as Processor
    participant VE as VisionEncoder
    participant L as MainLLM
    participant G as GenTokens
    participant S as SelectedStates
    participant C as Connector
    participant B as ByT5
    participant D as Diffusion
    participant I as Image

    U->>P: prompt and optional image
    P-->>L: input ids and pixel values

    opt image input
        L->>VE: extract image feature
        VE-->>L: image embeds
    end

    L->>G: append multiscale generation tokens
    G-->>L: expanded ids and gen mask
    L->>L: forward hidden states
    L-->>S: select gen-token states
    S-->>C: llm states
    C-->>D: condition embeds

    opt extra text control
        B-->>D: ByT5 embeds
    end

    D-->>I: generated image
```

Note: the Talker branch is loaded alongside the main model, but TTS generation does not execute through the main MoE LLM path.

## 4. Talker Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as SpeakerExtractor
    participant T as Talker
    participant Q as Qwen2
    participant C as CFM_DiT
    participant A as Aggregator
    participant H as StopHead
    participant V as AudioVAE
    participant W as Waveform

    Note over T,V: Optional side branch loaded with load_talker=True; no main MoE execution edge in TTS path

    U->>S: prompt wav
    S-->>T: speaker embedding
    U->>T: prompt text
    T->>Q: prefill embeds
    loop autoregressive audio
        Q-->>C: last hidden state
        C-->>A: next latent
        A-->>Q: next embeds
        Q-->>H: stop score
    end
    C-->>V: acoustic latents
    V-->>W: waveform
    W-->>U: audio
```

## 5. Performance Hotspots

```mermaid
flowchart LR
    A[Processor] --> B[Vision Encoder]
    A --> C[Audio Encoder]
    B --> D[Projectors Prompt Wrap]
    C --> D
    D --> E[MoE LLM with routing]
    E --> F1[Text Understanding]

    E -. image gen .-> G[Selected LLM States]
    G --> H[Connector]
    H --> I[Diffusion]
    I --> F2[Image Output]

    T1[Talker Qwen2 side branch] --> T2[CFM DiT]
    T2 --> T3[AudioVAE]
    T3 --> F3[Audio Output]
```

Note: the hotspot diagram lists Talker separately on purpose; it is not a runtime child stage of the main MoE backbone.

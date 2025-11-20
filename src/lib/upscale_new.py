import ffmpeg

def build_filter_graph(
    in_stream,
    src_w: int,
    src_h: int,
    scale_factor: float = 3.0,
    keep_ar: bool = True,
    # --- 추가 옵션 ---
    preset: str = "balanced",        # "fast" | "balanced" | "quality"
    out_pix_fmt: str = "yuv420p",      # OpenCV/파이프는 bgr24, 인코딩은 yuv420p
    use_zscale: bool = False,        # ffmpeg에 zscale 있을 때만 True
    high_bit_depth: bool = False,    # zscale+npl=16 사용 (품질↑, 속도↓)
    edge_mask: bool = False,         # 엣지 보호 샤픈 마스크
):
    """
    입력 스트림에 업스케일 필터를 적용하여 (필터 스트림, out_w, out_h) 반환.
    - 내부는 yuv444(옵션:16bit)에서 처리 → 마지막에 out_pix_fmt로 변환
    - preset으로 속도/품질 선택
    """

    # 0) 출력 해상도 (짝수 보정)
    if keep_ar:
        out_w = int(round(src_w * scale_factor))
        out_h = int(round(src_h * scale_factor))
    else:
        out_w = int(round(src_w * scale_factor))
        out_h = int(round(src_h * scale_factor))
    out_w = max(2, (out_w // 2) * 2)
    out_h = max(2, (out_h // 2) * 2)

    # 1) 내부 포맷 승격: yuv444 (+16bit 옵션)
    s = in_stream
    if use_zscale and high_bit_depth:
        # TV→FULL 확장까지 갈 경우: rangein='tv', range='full' 추가 고려
        s = s.filter('zscale', npl=16).filter('format', 'yuv444p16le')
    else:
        # 16bit가 안 되면 8bit 4:4:4라도 확보
        try:
            s = s.filter('format', 'yuv444p16le' if high_bit_depth else 'yuv444p')
        except Exception:
            s = s.filter('format', 'yuv444p')

    # 2) 프리셋별 전처리 + 스케일러 선택
    #    - swscale flags: fast_bilinear, bilinear, bicubic, bicublin, lanczos, spline, accurate_rnd, full_chroma_int
    if preset == "fast":
        # 최대 속도: 약한 노이즈면 전처리 생략, 스케일은 빠르게
        scale_flags = 'fast_bilinear+full_chroma_int'
        do_deband = False
        do_denoise = False
        sharpen_kind = None  # 샤픈 생략(속도)
    elif preset == "balanced":
        # 균형: hqdn3d 약, 스케일러 란초즈, 약한 샤픈
        scale_flags = 'lanczos+accurate_rnd+full_chroma_int'
        do_deband = False
        do_denoise = True
        sharpen_kind = "unsharp"
    elif preset == "quality":
        # 품질 우선: spline36(또는 zscale + spline), 디밴딩 옵션, 샤픈은 마스크 하에서
        scale_flags = 'spline+accurate_rnd+full_chroma_int'
        do_deband = True
        do_denoise = True
        sharpen_kind = "unsharp"  # cas 있으면 바꿔도 됨
    else:
        raise ValueError("preset must be one of: fast | balanced | quality")

    # 3) (옵션) 디노이즈 (너무 세게 하면 스머어) - 하드한 소스면 값 키움
    if do_denoise:
        # 안전한 기본값: 공간/시간 모두 약~중간
        s = s.filter('hqdn3d', 1.2, 1.2, 4, 4)

    # 4) 업스케일
    if use_zscale:
        # zscale로 고품질 보간 (filter: lanczos/spline36)
        if 'spline' in scale_flags:
            base = s.filter('zscale', w=out_w, h=out_h, filter='spline36')
        elif 'lanczos' in scale_flags:
            base = s.filter('zscale', w=out_w, h=out_h, filter='lanczos')
        else:
            base = s.filter('zscale', w=out_w, h=out_h, filter='bilinear')
    else:
        base = s.filter('scale', out_w, out_h, flags=scale_flags)

    # 5) 샤픈 (엣지 보호 마스크 선택 가능)
    core = base
    if sharpen_kind:
        if edge_mask:
            # 엣지 보호용 마스크 생성
            mask = (
                s.filter('edgedetect', low=0.05, high=0.15)
                .filter('format', 'gray')
                .filter('gblur', sigma=1.2)
                .filter('scale', out_w, out_h)
                .filter('negate')  # 엣지 보호(비엣지 영역 가중)
            )
            sharp = (base.filter('unsharp', 5, 5, 0.3, 3, 3, 0.3)
                        if sharpen_kind == "unsharp"
                        else base)
            core = ffmpeg.filter([base, sharp, mask], 'maskedmerge')
        else:
            core = base.filter('unsharp', 5, 5, 0.3, 3, 3, 0.3)

    # 6) (옵션) 디밴딩
    if do_deband:
        try:
            core = core.filter('deband', range=16, blur=0.5, thresh=64)
        except Exception:
            pass  # 빌드에 없으면 스킵

    # 7) 최종 출력 포맷 (raw 파이프/OpenCV면 bgr24, 인코딩이면 yuv420p)
    try:
        core = core.filter('format', out_pix_fmt)
    except Exception:
        # 안전장치: 지원 안 하면 bgr24로
        core = core.filter('format', 'bgr24')

    return core, out_w, out_h

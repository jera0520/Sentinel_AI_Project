import ffmpeg


def build_filter_graph(
    in_stream,
    src_w: int,
    src_h: int,
    scale_factor: float = 2.0,
    keep_ar: bool = True,
):
    """
    입력 스트림에 업스케일 필터를 적용하여 (필터 스트림, out_w, out_h)를 반환합니다.
    - scale_factor: 기본 x2 업스케일
    - keep_ar: True면 입력 종횡비 유지 (반올림 시 짝수 보정)
    - 필터 실패 가능성을 최소화하기 위해 기본 'scale'만 사용
    """
    # 출력 해상도 계산 (짝수 보정)
    if keep_ar:
        out_w = int(round(src_w * scale_factor))
        out_h = int(round(src_h * scale_factor))
    else:
        out_w = int(round(src_w * scale_factor))
        out_h = int(round(src_h * scale_factor))
    # 짝수 보정 및 하한 보정
    out_w = max(2, (out_w // 2) * 2)
    out_h = max(2, (out_h // 2) * 2)

    # 최소 필터 체인: format → scale → 최종 bgr24
    stream = in_stream
    stream = stream.filter('format', 'yuv444p')
    ''' 
    upscale 옵션
    빠른 업스케일 : bicublin+accurate_rnd+full_chroma_int
    링잉 최소화: spline+accurate_rnd+full_chroma_int
    샤프강조: lanczos+accurate_rnd+full_chroma_int
    '''
    stream = stream.filter('scale', out_w, out_h, flags='lanczos+accurate_rnd+full_chroma_int')
    stream = stream.filter('format', 'yuv420p')

    return stream, out_w, out_h
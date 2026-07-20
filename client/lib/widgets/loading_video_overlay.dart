import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

/// ????? ???????? ? ??????????? ?????.
/// ?????????? H.264 MP4 (zagruuuuzka.mp4), ??????? ??????????????
/// ???????????? video_player ?? Android ? iOS ? ????????? ??????????????.
class LoadingVideoOverlay extends StatefulWidget {
  final bool isLoading;

  const LoadingVideoOverlay({super.key, required this.isLoading});

  @override
  State<LoadingVideoOverlay> createState() => _LoadingVideoOverlayState();
}

class _LoadingVideoOverlayState extends State<LoadingVideoOverlay> {
  late final VideoPlayerController _controller;
  bool _initialized = false;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.asset('assets/zagruuuuzka.mp4')
      ..setLooping(true)
      ..initialize().then((_) {
        if (!mounted) return;
        setState(() => _initialized = true);
        if (widget.isLoading) _controller.play();
      });
  }

  @override
  void didUpdateWidget(covariant LoadingVideoOverlay oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_initialized) return;
    if (widget.isLoading && !_controller.value.isPlaying) {
      _controller.seekTo(Duration.zero);
      _controller.play();
    } else if (!widget.isLoading && _controller.value.isPlaying) {
      _controller.pause();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Offstage(
      offstage: !widget.isLoading,
      child: Container(
        color: Colors.black.withValues(alpha: 0.6),
        width: double.infinity,
        height: double.infinity,
        child: _initialized
            ? FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: _controller.value.size.width,
                  height: _controller.value.size.height,
                  child: VideoPlayer(_controller),
                ),
              )
            : const Center(
                child: CircularProgressIndicator(color: Color(0xFFF5C518)),
              ),
      ),
    );
  }
}

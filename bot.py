import logging
import os
import uuid
from pathlib import Path
import shutil
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
from rembg import new_session, remove
import onnxruntime as ort 
from PIL import Image
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

TELEGRAM_BOT_TOKEN = "" # Enter your tg bot token 
TEMP_DIR = Path("./temp_bot_files")


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("rembg").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


user_backgrounds = {}


def initialize_rembg_session():
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    # 'u2net', 'u2netp', 'isnet-general-use' 'u2net_human_seg'
    model_name = "u2net_human_seg"
    logging.getLogger("onnxruntime").setLevel(logging.INFO) 

    try:
        logger.info(f"Attempting to initialize rembg session (model: {model_name}) with providers: {providers}")

        session = new_session(model_name=model_name, providers=providers)
        logger.info(f"Successfully initialized rembg session. ONNX Runtime selected the best available provider (attempted DirectML first).")
        

        return session
    except Exception as e:
        
        logger.error(f"Failed to initialize ONNX Runtime session with DirectML attempt: {e}", exc_info=True)
        logger.warning("Falling back to CPU-only session for rembg.")
        
        try:
            
            session = new_session(model_name=model_name, providers=['CPUExecutionProvider'])
            logger.info("Initialized rembg session using CPU provider as fallback.")
            return session
        except Exception as fallback_e:
            logger.critical(f"FATAL: Could not initialize rembg session even with CPU fallback: {fallback_e}", exc_info=True)
            return None 


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n\n"
        "Я могу заменить фон на твоих видео или видеосообщениях (кружочках).\n\n"
        "<b>Как использовать:</b>\n"
        "1. Отправь мне картинку, которая будет новым фоном.\n"
        "2. Отправь видео или видеосообщение, в котором нужно заменить фон.\n\n"
        "Я обработаю видео и пришлю результат в том же формате (видео -> видео, кружок -> кружок)."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

# --- Photo Handler (handle_photo - unchanged) ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo_file = await update.message.photo[-1].get_file()
    user_temp_dir = TEMP_DIR / str(user_id)
    user_temp_dir.mkdir(parents=True, exist_ok=True)
    bg_filename = f"background_{uuid.uuid4()}.png"
    bg_path = user_temp_dir / bg_filename

    try:
        await photo_file.download_to_drive(bg_path)
        logger.info(f"Фон сохранен для пользователя {user_id} по пути: {bg_path}")
        if user_id in user_backgrounds and user_backgrounds[user_id].exists():
             try:
                 os.remove(user_backgrounds[user_id])
                 logger.info(f"Старый фон для {user_id} удален.")
             except OSError as e:
                 logger.error(f"Ошибка удаления старого фона для {user_id}: {e}")
        user_backgrounds[user_id] = bg_path

        await update.message.reply_text("Отлично! Фон сохранен. Теперь отправь мне видео или видеосообщение.")

    except Exception as e:
        logger.error(f"Ошибка при сохранении фона для {user_id}: {e}")
        await update.message.reply_text("Не удалось сохранить фон. Попробуй еще раз.")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает видео или видеосообщение."""
    user_id = update.effective_user.id
    message = update.message

    if 'rembg_session' not in context.bot_data or context.bot_data['rembg_session'] is None:
         logger.error("Rembg ONNX session not initialized. Cannot process video.")
         await message.reply_text("Ошибка: Не удалось инициализировать модель обработки. Попробуйте перезапустить бота или обратитесь к администратору.", quote=True)
         return
    rembg_session = context.bot_data['rembg_session'] # Get the session

    if user_id not in user_backgrounds or not user_backgrounds[user_id].exists():
        await message.reply_text("Сначала отправь мне картинку для фона!")
        return

    is_video_note = bool(message.video_note)
    video_file_id = message.video_note.file_id if is_video_note else message.video.file_id
    if is_video_note:
        original_width = message.video_note.length
        original_height = message.video_note.length
    else:
        original_width = getattr(message.video, 'width', 640)
        original_height = getattr(message.video, 'height', 480)
        if not original_width or not original_height:
             logger.warning(f"Video dimensions missing for video from user {user_id}. Using defaults (640x480).")
             original_width = 640
             original_height = 480

    target_size = (original_width, original_height)

    try:
        video_file = await context.bot.get_file(video_file_id)
    except Exception as e:
         logger.error(f"Ошибка получения файла ({video_file_id}) для {user_id}: {e}")
         await message.reply_text("Не удалось скачать видеофайл. Возможно, он слишком большой или удален.")
         return

    request_id = uuid.uuid4()
    request_temp_dir = TEMP_DIR / str(user_id) / str(request_id)
    request_temp_dir.mkdir(parents=True, exist_ok=True)

    input_video_path = request_temp_dir / f"input_{request_id}.mp4"
    output_video_path = request_temp_dir / f"output_{request_id}.mp4"
    bg_path = user_backgrounds[user_id]

    processing_msg = await message.reply_text("Начинаю обработку видео... Это может занять некоторое время ⏳", quote=True)

    try:
        await video_file.download_to_drive(input_video_path)
        logger.info(f"Видео скачано для {user_id} в {input_video_path}")

        
        success = await process_video_background(
            input_video_path,
            bg_path,
            output_video_path,
            target_size,
            is_video_note,
            rembg_session 
        )

        if success:
            logger.info(f"Видео обработано для {user_id}, результат: {output_video_path}")
            try:
                if output_video_path.stat().st_size > 0:
                    with open(output_video_path, 'rb') as video_send_file:
                        if is_video_note:
                            await message.reply_video_note(video_note=video_send_file)
                        else:
                            await message.reply_video(video=video_send_file, supports_streaming=True, width=original_width, height=original_height) 
                    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=processing_msg.message_id)
                else:
                    error_msg = "Обработанный файл видеосообщения пустой." if is_video_note else "Обработанный файл видео пустой."
                    logger.error(f"{error_msg} для {user_id} по пути {output_video_path}")
                    raise ValueError(error_msg)

            except ValueError as ve:
                logger.error(f"ValueError {user_id}: {ve}")
                await processing_msg.edit_text(f"Произошла ошибка: {ve}. Попробуйте другое видео.")
            except Exception as send_error:
                logger.error(f"Ошибка отправки обработанного видео для {user_id}: {send_error}", exc_info=True)
                await processing_msg.edit_text(f"Не удалось отправить результат. Возможно, файл слишком большой, поврежден или произошла сетевая ошибка.")

        else:
            logger.error(f"Обработка видео не удалась для {user_id} (файл {input_video_path})")
            await processing_msg.edit_text("К сожалению, во время обработки видео произошла ошибка. Попробуйте другое видео или фон.")

    except Exception as e:
        logger.exception(f"Критическая ошибка при обработке видео для {user_id}: {e}")
        try:
            await processing_msg.edit_text("Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")
        except Exception as edit_err:
             logger.warning(f"{edit_err}")

    finally:
        if request_temp_dir.exists():
            try:
                shutil.rmtree(request_temp_dir)
                logger.info(f"Временная папка {request_temp_dir} удалена.")
            except Exception as cleanup_error:
                logger.error(f"Ошибка удаления временной папки {request_temp_dir}: {cleanup_error}")



async def process_video_background(
    video_path: Path,
    bg_path: Path,
    output_path: Path,
    target_size: tuple,
    is_video_note: bool,
    rembg_session: ort.InferenceSession
) -> bool:

    try:

        background_img = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if background_img is None:
            logger.error(f"Не удалось загрузить фон: {bg_path}")
            return False

        background_resized = cv2.resize(background_img, target_size, interpolation=cv2.INTER_AREA)
        background_resized_rgba = cv2.cvtColor(background_resized, cv2.COLOR_BGR2RGBA)

 
        video_clip = VideoFileClip(str(video_path))
        original_audio = video_clip.audio
        fps = video_clip.fps or 25 
        logger.info(f"Видео: {video_path.name}, Size: {video_clip.size}, FPS: {fps}, Аудио: {'есть' if original_audio else 'нет'}")

        if video_clip.size != list(target_size):
            logger.warning(f"Video clip size {video_clip.size} differs from target size {target_size}. Resizing frames.")
            
        processed_frames = []
        frame_count = 0
        for frame_rgb in video_clip.iter_frames(fps=fps, dtype='uint8'):
            frame_count += 1
            frame_pil = Image.fromarray(frame_rgb)
            try:
                 frame_rgba_pil = remove(frame_pil, session=rembg_session, alpha_matting=False)
            except Exception as rembg_err:
                 logger.error(f"Error during rembg processing on frame {frame_count}: {rembg_err}", exc_info=True)
                 continue

            frame_rgba = np.array(frame_rgba_pil)
            current_h, current_w = frame_rgba.shape[:2]
            target_w, target_h = target_size

            if current_w != target_w or current_h != target_h:
                frame_rgba_resized = cv2.resize(frame_rgba, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                frame_rgba_resized = frame_rgba
            alpha_channel = frame_rgba_resized[:, :, 3] / 255.0
            alpha_mask_fg = np.stack([alpha_channel] * 4, axis=-1)
            alpha_mask_bg = 1.0 - alpha_mask_fg 
            foreground = frame_rgba_resized.astype(np.float32) * alpha_mask_fg
            background = background_resized_rgba.astype(np.float32) * alpha_mask_bg
            composite_frame_rgba = cv2.add(foreground, background).clip(0, 255).astype(np.uint8)
            final_frame_rgb = cv2.cvtColor(composite_frame_rgba, cv2.COLOR_RGBA2RGB)
            processed_frames.append(final_frame_rgb)

        logger.info(f"Обработано {len(processed_frames)} / {frame_count} кадров.")

        if not processed_frames:
            logger.error("Не было успешно обработано ни одного кадра.")
            video_clip.close()
            if original_audio: original_audio.close()
            return False

        new_video_clip = ImageSequenceClip(processed_frames, fps=fps)

        if original_audio:
            logger.info("Добавляю оригинальное аудио.")
            new_video_clip = new_video_clip.set_audio(original_audio)
        else:
             logger.info("Аудио в исходном видео не найдено.")


        codec = "libx264"
        audio_codec = "aac"
        preset = 'ultrafast'
        threads = os.cpu_count() or 4 


        logger.info(f"Начинаю запись файла: {output_path} с кодеком {codec}, аудио {audio_codec}, пресет {preset}, threads={threads}")
        try:
            new_video_clip.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec=audio_codec,
                preset=preset,
                threads=threads,
                logger='bar'
            )
        except Exception as write_error:
             logger.error(f"Ошибка при записи видеофайла с кодеком {codec}: {write_error}", exc_info=True)
             if codec != "libx264":
                 codec = "libx264"
                 preset = 'ultrafast'
                 new_video_clip.write_videofile(
                     str(output_path),
                     codec=codec,
                     audio_codec=audio_codec,
                     preset=preset,
                     threads=threads,
                     logger='bar'
                 )
             else:
                 raise write_error

        new_video_clip.close()
        video_clip.close()
        if original_audio:
             original_audio.close()

        logger.info(f"Файл успешно сохранен: {output_path}")
        return True

    except Exception as e:
        logger.exception(f"Ошибка в process_video_background: {e}")
        if 'video_clip' in locals() and video_clip: video_clip.close()
        if 'original_audio' in locals() and original_audio: original_audio.close()
        if 'new_video_clip' in locals() and new_video_clip: new_video_clip.close()
        return False

def main():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing rembg ONNX session...")
    rembg_session = initialize_rembg_session()
    if rembg_session is None:
        logger.critical("Could not initialize rembg session. Bot cannot function properly.")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data['rembg_session'] = rembg_session
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))
    application.add_handler(MessageHandler((filters.VIDEO | filters.VIDEO_NOTE) & ~filters.COMMAND, handle_video))


    logger.info("Бот запускается...")
    try:
        application.run_polling()
    finally:
        logger.info("Бот останавливается...")
        if 'rembg_session' in application.bot_data:
            application.bot_data['rembg_session'] = None
            logger.info("Rembg session cleared.")


        try:
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
                logger.info(f"Временная папка {TEMP_DIR} очищена при остановке.")
        except Exception as cleanup_error:
            logger.error(f"Ошибка очистки временной папки при остановке: {cleanup_error}")


if __name__ == "__main__":
    main()
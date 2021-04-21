from django.views import View
from django.http import JsonResponse
from pose.pose_engine import Pose_Check
from django.core.files.storage import default_storage
from pose_check.settings import MEDIA_ROOT


class PoseCheckView(View):
    def post(self, request):
        video_file = request.FILES['file']
        default_storage.save(MEDIA_ROOT+video_file.name, video_file)

        pose_json = Pose_Check.pose_run(self, MEDIA_ROOT, video_file.name)

        return JsonResponse({'pose_json' : pose_json}, status=200)
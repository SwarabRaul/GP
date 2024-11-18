// 21BAI1225 - Swarab Raul

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class moveScene : MonoBehaviour
{
    public void MoveScene(int SceneID)
    {
        SceneManager.LoadScene(SceneID);
    }
}
